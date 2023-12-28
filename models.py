import keras.layers
import numpy as np
from keras.models import *
from keras.layers import Input,Conv2D,Activation,Lambda,Subtract,concatenate,Add,GlobalAveragePooling2D,PReLU,Dense,Multiply,Reshape,BatchNormalization
import keras.backend as K
import tensorflow as tf
import os

def PAGDNet(): #original format def PAGDNet(), data is used to obtain the reshape of input data
    inpt = Input(shape=(None,None,1)) #if the image is 3, it is color image. If the image is 1, it is gray image
    #layer 1
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    s = x
    f = x
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,1), strides=(1,1), padding='same')(s)
        x = Conv2D(filters=64, kernel_size=(1,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=64, kernel_size=(3,1), strides=(1,1), padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(1,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        pixel = Dense(4, activation='relu', use_bias=False)(x)
        pixel = Dense(1, activation='sigmoid', use_bias=False)(pixel)
        pixel = Multiply()([x, pixel])
        s = Add()([pixel, s])
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(s)
    x = BatchNormalization()(x)
    o = Add()([x, f])
    o = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(o)  # gray is 1 color is 3
    z = Subtract()([inpt, o])
    model = Model(inputs=inpt, outputs=z)
    # model.summary()
    return model