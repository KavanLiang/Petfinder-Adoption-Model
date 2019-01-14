from Models.BlockLayers import dense_block, transition_block
from keras.layers import Dense, BatchNormalization, Input, ReLU, GlobalAveragePooling2D, Softmax, Dropout, Embedding, \
    concatenate
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.models import Model
from keras.regularizers import l1_l2

from keras.applications import DenseNet201


def mixed_model(input_shapes, num_conv_layers, num_dense_layers, filters, growth_rate, embedded_layer_dim, dropout=0, use_dropout=False, reduction=0.0,
                reg_param=0.0):
    compression = 1.0 - reduction

    input_1 = Input(shape=input_shapes[0])
    embedded_dummy = Dense(embedded_layer_dim, kernel_regularizer=l1_l2(reg_param, reg_param), activation='relu')(input_1)

    input_2 = Input(shape=input_shapes[1])  # continuous

    input_3 = Input(shape=input_shapes[2])  # image

    x = Conv2D(filters, (7, 7), strides=(2, 2), padding='same')(input_3)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    for i in range(len(num_conv_layers) - 1):
        x, filters = dense_block(x, filters, num_conv_layers[i], growth_rate=growth_rate, reg_param=reg_param)
        x = transition_block(x, filters, compression_ratio=compression, reg_param=reg_param)
        filters = int(filters * compression)

    x, filters = dense_block(x, filters, num_conv_layers[-1], growth_rate=growth_rate, reg_param=reg_param)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)

    x = concatenate([x, input_2, embedded_dummy])

    for i in num_dense_layers:
        x = Dense(i, activation='relu', kernel_regularizer=l1_l2(reg_param, reg_param))(x)
        if use_dropout:
            x = Dropout(dropout)(x)

    x = Dense(5, kernel_regularizer=l1_l2(reg_param, reg_param))(x)
    x = Softmax()(x)

    return Model(inputs=[input_1, input_2, input_3], outputs=[x])

def mixed_dense_model(input_shapes, embedding_dim, dense_dims, reg_param=0.0, dropout=0.0, use_dropout=False):
    input_1 = Input(shape=input_shapes[0])
    embedded_dummy = Dense(embedding_dim, kernel_regularizer=l1_l2(reg_param, reg_param), activation='relu')(input_1)

    input_2 = Input(shape=input_shapes[1])  # continuous

    base_model = DenseNet201(include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = concatenate([x, input_2, embedded_dummy])
    for i in dense_dims:
        x = Dense(i, kernel_regularizer=l1_l2(reg_param, reg_param), activation='relu')(x)
        if use_dropout:
            x = Dropout(dropout)(x)
    x = Dense(5)(x)
    x = Softmax()(x)

    return Model(inputs=[input_1, input_2, base_model.input], outputs=x)
