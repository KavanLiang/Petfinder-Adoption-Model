import sys
import os
import re
import numpy as np
from glob import glob
import pickle

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import keras

from Models import MixedModel

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from DataFormatter import mkdir
from keras.utils import Sequence, to_categorical

from keras.applications.densenet import preprocess_input

from PIL.Image import LANCZOS

sys.stderr = stderr

# Approximate
NUM_TRAINING_EXAMPLES = 43865
NUM_VALIDATION_EXAMPLES = 14787

CLASS_WEIGHTS = {0: 11.973837209302326, 1: 1.435191637630662, 2: 1.0, 3: 1.0892022917584838, 4: 1.174508126603935}

class DataGenerator(Sequence):

    def __init__(self, directory_path, batch_size, input_shapes, shuffle=True, use_imagenet_preprocessing=False):
        """
        Preconditions: label directories and classes are 0-ordinal

        :param directory_path: relative path
        :param input_shapes: list of input shapes, last one is the image shape (images last)
        :param batch_size: batch size
        :param shuffle: whether or not to shuffle the dataset
        """
        self.use_imagenet_preprocessing = use_imagenet_preprocessing
        self.img_data_generator = ImageDataGenerator(rotation_range=0.45,
                                                     horizontal_flip=True) if self.use_imagenet_preprocessing else ImageDataGenerator(
            rotation_range=0.45, horizontal_flip=True, rescale=1. / 255.)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.input_shapes = input_shapes

        # generate file paths and labels
        self.path_list = glob(f'{directory_path}\\*\\*')

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        path_list_temp = [self.path_list[k] for k in indexes]

        X, y = self._data_generation(path_list_temp)

        return X, y

    def _data_generation(self, path_list_temp):
        """Generates data containing batch_size samples"""
        X = [np.empty((self.batch_size, *shape)) for shape in self.input_shapes]
        y = np.empty(self.batch_size, dtype='float32')

        for i, path in enumerate(path_list_temp):
            with open(path, 'rb') as file:
                data = pickle.load(file)
                data[1] = data[1].astype('float32')
                data[1] /= (np.max(np.abs(data[1]), axis=0) + 1e-5)  # prevent divide by 0
                data[-1] = self.img_data_generator.random_transform(img_to_array(
                    array_to_img(data[-1]).resize(self.input_shapes[-1][:-1])))
                if self.use_imagenet_preprocessing:
                    data[-1] = preprocess_input(data[-1])
                for j in range(len(self.input_shapes)):
                    X[j][i,] = data[j]
                y[i] = np.float32(re.search('\d', path).group(0))

        return X, to_categorical(y, num_classes=5)

    def on_epoch_end(self):
        """shuffles indexes after each epoch"""
        self.indexes = np.arange(len(self.path_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_mixed_model(use_latest=False, new_dir=True):
    input_shapes = [(369,), (9,), (200, 200, 3)]
    num_conv_layers = [4, 7, 24, 7]
    num_dense_layers = [324, 324]
    filters = 32
    growth_rate = 32
    embedded_layer_dim = 128
    dropout = 0.35
    use_dropout = True
    reg_param = 1e-6
    batch_size = 32

    init_lr = 1e-3
    init_epoch = 0

    model_name = f'MixNet-ADAM-{init_lr}-BATCH_SIZE-{batch_size}-IMAGE_DIM-{input_shapes[-1]}-l1_l2-REG-{reg_param}-filters-{filters}-GR-{growth_rate}-dropout-{dropout}-convlayers-(' + ', '.join(
        [str(i) for i in num_conv_layers]) + f')-embedding-dim-{embedded_layer_dim}-num_dense_layers-(' + ', '.join(
            [str(i) for i in num_dense_layers]) + ')'

    if new_dir:
        mkdir(f'SLink/PetfinderModels/{model_name}/')

    if use_latest:
        latest = max(glob(f'SLink/PetfinderModels/{model_name}/*.hdf5'), key=os.path.getctime)
        model = keras.models.load_model(latest)
        init_epoch = int(
            re.search('[0-9]+', re.search('weights\.[0-9]+-', latest).group(0)).group(
                0))
    else:
        model = MixedModel.mixed_model(input_shapes, num_conv_layers, num_dense_layers, filters, growth_rate, embedded_layer_dim,
                                       dropout=dropout,
                                       use_dropout=use_dropout, reg_param=reg_param)
        model.compile(optimizer=optimizers.Adam(lr=init_lr, amsgrad=True), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

    train_generator = DataGenerator('SLink/PetfinderDatasets/Train', batch_size, input_shapes=input_shapes)
    val_generator = DataGenerator('SLink/PetfinderDatasets/Val', batch_size, input_shapes=input_shapes)

    callbacks = [
        ModelCheckpoint(f'SLink/PetfinderModels/{model_name}/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5,
                        verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, cooldown=10),
        TensorBoard(log_dir=f'logs/{model_name}')
    ]

    print(model.summary())

    model.fit_generator(generator=train_generator, validation_data=val_generator,
                        validation_steps=NUM_VALIDATION_EXAMPLES // batch_size,
                        verbose=1,
                        epochs=10000,
                        callbacks=callbacks,
                        steps_per_epoch=NUM_TRAINING_EXAMPLES // (batch_size * 4),
                        initial_epoch=init_epoch,
                        class_weight=CLASS_WEIGHTS,
                        use_multiprocessing=True,
                        workers=12)

if __name__ == '__main__':
    train_mixed_model()
