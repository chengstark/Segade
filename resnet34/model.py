import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, \
    Lambda
from keras.models import Model

AUDIO_LENGTH = 7201


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    resnet 34 identity block
    :param input_tensor: input tensor
    :type input_tensor: tf.Tensor
    :param kernel_size: kernel size
    :type kernel_size: int
    :param filters: filters
    :type filters: int
    :param stage: stage
    :type stage: int
    :param block: block
    :type block: int
    :return: output
    :rtype: tf.Tensor
    """
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    # up-sample from the activation maps.
    # otherwise it's a mismatch. Recommendation of the authors.
    # here we x2 the number of filters.
    # See that as duplicating everything and concatenate them.
    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def resnet_34(num_classes=2):
    """
    Generate renset 34 model
    :param num_classes: number of classes
    :type num_classes: int
    :return: resnet 34 model
    :rtype: tf.keras.Model
    """
    inputs = Input(shape=[AUDIO_LENGTH, 1])

    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling1D(pool_size=4, strides=None)(x)
    print(x.shape)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)
    print(x.shape)

    for i in range(4):
        x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(6):
        x = identity_block(x, kernel_size=3, filters=192, stage=3, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)
    print(x.shape)
    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=384, stage=4, block=i)

    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    m = Model(inputs, x, name='resnet34')
    return m


if __name__ == '__main__':
    resnet_34()
