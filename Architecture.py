# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:09:18 2019

@author: jiaoyi2


simplified VGG 16
"""


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class DeepConvNetwork:
    def build(sizeOfInput,numOfClass,dropRate=0.5,regPara=0.01,weightsPath=None):
        model=Sequential()
        
        # Layer 1 Input Size = 48*48*3
        model.add(Conv2D(16, (3,3), input_shape=sizeOfInput,
                           padding='same', kernel_regularizer=l2(regPara)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        # Layer 2 Input Size = 48*48*16
        model.add(Conv2D(16,(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        # Layer 3 Input Size = 24*24*16
        model.add(Conv2D(32,(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        
        # Layer 4 Input Size = 24*24*32
        model.add(Conv2D(32,(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        # Layer 5 Input Size = 12*12*32
        model.add(Conv2D(64,(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        # Layer 6 Input Size = 6*6*64
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropRate))
        
        # Layer 7 
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropRate))
        
        # Layer 8
        model.add(Dense(numOfClass))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        
        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model
