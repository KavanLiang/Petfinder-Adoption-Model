from keras.applications import DenseNet169
from keras.layers import Dense, BatchNormalization, Input, ReLU, GlobalAveragePooling2D, Softmax, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.models import Model
from keras.regularizers import l1_l2

from Models.BlockLayers import dense_block, transition_block


def conv_adopt_model(input_dim, num_layers, filters, growth_rate, reduction=0.0, reg_param=0.0):
    compression = 1.0 - reduction

    input_layer = Input(shape=(input_dim, input_dim, 3))

    x = Conv2D(filters, (7, 7), strides=(2, 2), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    for i in range(len(num_layers) - 1):
        x, filters = dense_block(x, filters, num_layers[i], growth_rate=growth_rate, reg_param=reg_param)
        x = transition_block(x, filters, compression_ratio=compression, reg_param=reg_param)
        filters = int(filters * compression)

    x, filters = dense_block(x, filters, num_layers[-1], growth_rate=growth_rate, reg_param=reg_param)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(5, kernel_regularizer=l1_l2(reg_param, reg_param))(x)
    x = Softmax()(x)

    return Model(inputs=[input_layer], outputs=[x])


def transfer_learning_densenet():
    base_model = DenseNet169(include_top=False)
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(5, activation='softmax')(x)

    return Model(inputs=[base_model.input], outputs=[x])
