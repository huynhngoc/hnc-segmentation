import gc
from itertools import product
import shutil
from deoxys.data.preprocessor import BasePreprocessor
from deoxys_image.patch_sliding import get_patch_indice, get_patches, \
    check_drop
import h5py
from tensorflow import image
from tensorflow.keras.layers import Input, concatenate, Lambda, \
    Add, Activation, Multiply
from tensorflow.keras.models import Model as KerasModel
import numpy as np
import tensorflow as tf
from deoxys.model.callbacks import PredictionCheckpoint
from deoxys.loaders.architecture import BaseModelLoader
from deoxys.experiment import Experiment
from deoxys.experiment.postprocessor import DefaultPostProcessor
from deoxys.utils import file_finder, load_json_config
from deoxys.customize import custom_architecture, custom_datareader, custom_layer
from deoxys.loaders import load_data
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator
from deoxys.model.layers import layer_from_config
# from tensorflow.python.ops.gen_math_ops import square
import tensorflow_addons as tfa
from deoxys.model.losses import Loss, loss_from_config
from deoxys.customize import custom_loss, custom_preprocessor
from deoxys.data import ImageAugmentation2D
from elasticdeform import deform_random_grid
import os

multi_input_layers = ['Add', 'AddResize', 'Concatenate', 'Multiply', 'Average']
resize_input_layers = ['Concatenate', 'AddResize']


@custom_preprocessor
class CroppedMask3D(BasePreprocessor):
    def __init__(self, channel=-1, size=128):
        self.channel = channel
        self.size = size
        self.left = size // 2
        self.right = size - self.left

    def _get_bounding(self, mask, axis, max_size):
        args = np.argwhere(mask.sum(axis=axis) > 0).flatten()
        middle = (args.max() - args.min()) // 2
        left, right = middle - self.left, middle + self.right
        if left < 0:
            left = 0
            right = self.size
        if right > max_size:
            right = max_size
            left = right - self.size

        return left, right

    def transform(self, images, targets):
        masks = images[..., self.channel]
        shape = masks.shape[1:]
        new_images = []
        for i, mask in enumerate(masks):
            left_0, right_0 = self._get_bounding(
                mask, axis=(1, 2), max_size=shape[0])
            left_1, right_1 = self._get_bounding(
                mask, axis=(0, 2), max_size=shape[1])
            left_2, right_2 = self._get_bounding(
                mask, axis=(0, 1), max_size=shape[2])

            new_images.append(
                images[i][left_0: right_0, left_1: right_1, left_2: right_2]
            )

        return np.array(new_images), targets


@custom_layer
class InstanceNormalization(tfa.layers.InstanceNormalization):
    pass


@custom_layer
class AddResize(Add):
    pass


@custom_loss
class BinaryMacroFbetaLoss(Loss):
    def __init__(self, reduction='auto', name="binary_macro_fbeta",
                 beta=1, square=False):
        super().__init__(reduction, name)

        self.beta = beta
        self.square = square

    def call(self, target, prediction):
        eps = 1e-8
        target = tf.cast(target, prediction.dtype)

        true_positive = tf.math.reduce_sum(prediction * target)
        if self.square:
            target_positive = tf.math.reduce_sum(tf.math.square(target))
            predicted_positive = tf.math.reduce_sum(
                tf.math.square(prediction))
        else:
            target_positive = tf.math.reduce_sum(target)
            predicted_positive = tf.math.reduce_sum(prediction)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return 1 - fb_numerator / fb_denominator


@custom_loss
class FusedLoss(Loss):
    """Used to sum two or more loss functions.
    """

    def __init__(
            self, loss_configs, loss_weights=None,
            reduction="auto", name="fused_loss"):
        super().__init__(reduction, name)
        self.losses = [loss_from_config(loss_config)
                       for loss_config in loss_configs]

        if loss_weights is None:
            loss_weights = [1] * len(self.losses)
        self.loss_weights = loss_weights

    def call(self, target, prediction):
        loss = None
        for loss_class, loss_weight in zip(self.losses, self.loss_weights):
            if loss is None:
                loss = loss_weight * loss_class(target, prediction)
            else:
                loss += loss_weight * loss_class(target, prediction)

        return loss


class EnsemblePostProcessor(DefaultPostProcessor):
    def __init__(self, log_base_path='logs',
                 log_path_list=None,
                 map_meta_data=None, **kwargs):

        self.log_base_path = log_base_path
        self.log_path_list = []
        for path in log_path_list:
            merge_file = path + self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            if os.path.exists(merge_file):
                self.log_path_list.append(merge_file)
            else:
                print('Missing file from', path)

        # check if there are more than 1 to ensemble
        assert len(self.log_path_list) > 1, 'Cannot ensemble with 0 or 1 item'

        if map_meta_data:
            if type(map_meta_data) == str:
                self.map_meta_data = map_meta_data.split(',')
            else:
                self.map_meta_data = map_meta_data
        else:
            self.map_meta_data = ['patient_idx']

        # always run test
        self.run_test = True

    def ensemble_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        output_file = output_folder + self.PREDICT_TEST_NAME
        if not os.path.exists(output_file):
            print('Copying template for output file')
            shutil.copy(self.log_path_list[0], output_folder)

        print('Creating ensemble results...')
        y_preds = []
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                y_preds.append(hf['predicted'][:])

        with h5py.File(output_file, 'a') as mf:
            mf['predicted'][:] = np.mean(y_preds, axis=0)
        print('Ensembled results saved to file')

        return self

    def concat_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        # first check the template
        with h5py.File(self.log_path_list[0], 'r') as f:
            ds_names = list(f.keys())
        ds = {name: [] for name in ds_names}

        # get the data
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                for key in ds:
                    ds[key].append(hf[key][:])

        # now merge them
        print('creating merged file')
        output_file = output_folder + self.PREDICT_TEST_NAME
        with h5py.File(output_file, 'w') as mf:
            for key, val in ds.items():
                mf.create_dataset(key, data=np.concatenate(val, axis=0))


@custom_preprocessor
class ZScoreDensePreprocessor(BasePreprocessor):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, inputs, target=None):
        mean, std = self.mean, self.std
        if mean is None:
            mean = inputs.mean(axis=0)
            std = inputs.std(axis=0)
        else:
            mean = np.array(mean)
            std = np.array(std)
        std[std == 0] = 1

        return (inputs - mean)/std, target




@custom_preprocessor
class ElasticDeform(BasePreprocessor):
    def __init__(self, sigma=4, points=3):
        self.sigma = sigma
        self.points = points

    def transform(self, x, y):
        return deform_random_grid([x, y], axis=[(1, 2, 3), (1, 2, 3)],
                                  sigma=self.sigma, points=self.points)
