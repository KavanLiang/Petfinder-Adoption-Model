from keras.layers import Dense, BatchNormalization, Input, concatenate, ReLU, multiply, Reshape, Activation, \
    GlobalAveragePooling2D, Softmax, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.models import Model
from keras.regularizers import l1_l2

def se_block(x, compression_ratio=1.0 / 16.0, reg_param=0.0):
    '''
    A squeeze excitation block

    :param x: input block
    :param compression_ratio: ratio to scale the dense blocks
    :param reg_param: regularization parameter

    :return: input layer with squeeze excitation applied
    '''
    num_filters = x._keras_shape[-1]

    ret = GlobalAveragePooling2D()(x)
    ret = Reshape((1, 1, num_filters))(ret)
    ret = Dense(int(num_filters * compression_ratio), kernel_regularizer=l1_l2(reg_param, reg_param), )(ret)
    ret = ReLU()(ret)
    ret = Dense(num_filters, kernel_regularizer=l1_l2(reg_param, reg_param), )(ret)
    ret = Activation('sigmoid')(ret)

    return multiply([x, ret])


def transition_block(x, filters, compression_ratio=1.0, reg_param=0.0):
    '''
    Transition block to reduce featuremap size, combined with squeeze excitation implementation

    :param x: input block
    :param filters: number of filters
    :param compression_ratio: ratio to scale the number of filters
    :param reg_param: regularization parameter

    :return: transition block
    '''
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(int(filters * compression_ratio), (3, 3), padding='same',
               kernel_regularizer=l1_l2(reg_param, reg_param))(x)
    x = AveragePooling2D()(x)

    x = se_block(x, reg_param=reg_param)
    return x


def conv_block(x, filters, compression_ratio=1.0, reg_param=0.0):
    '''
    A convolutional block

    :param x: input layer
    :param filters: number of filters
    :param compression_ratio: the ratio to compress the number of filters
    :param reg_param: regularization parameter

    :return: the convolutional block
    '''
    x = BatchNormalization()(x)
    x = ReLU()(x)

    inter_channel = int(filters * compression_ratio)
    x = Conv2D(inter_channel, (1, 1), padding='same', kernel_regularizer=l1_l2(reg_param, reg_param))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l1_l2(reg_param, reg_param))(x)

    return x


def dense_block(x, filters, num_layers, growth_rate=0, reg_param=0.0, grow_filters=True, compression_ratio=1):
    '''
    A dense block with squeeze excitation implementation

    :param x: input layer
    :param filters: number of filters
    :param num_layers: the number of layers in the dense block
    :param growth_rate: the growth rate
    :param reg_param: regularization parameter
    :param grow_filters: whether or not to grow the number of filters in each successive convolutional block
    :param compression_ratio: the ratio of which to compress the filters in the convolutional and transition blocks

    :return: a dense block
    '''
    feature_list = [x]

    for i in range(num_layers):
        x = conv_block(x, growth_rate, compression_ratio=compression_ratio, reg_param=reg_param)
        feature_list.append(x)
        x = concatenate(feature_list)
        if grow_filters:
            filters += growth_rate
    x = se_block(x, reg_param=reg_param)
    return x, filters