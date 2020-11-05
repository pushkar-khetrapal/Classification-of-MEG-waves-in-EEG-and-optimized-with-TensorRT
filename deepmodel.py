import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, concatenate
from tensorflow.keras import metrics
from tensorflow.keras.layers import Flatten, Conv1D, AveragePooling1D, MaxPooling1D, BatchNormalization
from sklearn.metrics import confusion_matrix
import time
import tensorflow.compat.v1.keras.backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.compiler.tensorrt import trt_convert as tftrt
import copy
import time


def base_model(i):

    inputs = Input(shape=(33, 1), name='input_{}'.format(i))  

    x = Conv1D(filters=32, kernel_size=3)(inputs)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=3)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

def eeg_model():
    model_lis = []
    model_input = []
    model_output = []
    for i in range(10):   # creating a list of 10 different 1-D CNN models
        m = base_model(i)
        model_lis.append(m)
        model_output.append(m.output)
        model_input.append(m.input)
    
    combinedInput = concatenate(model_output)  ## concatenate all the models which made earlier

    # dense layers
    x = Dense(100, activation='relu')(combinedInput)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(3, activation='softmax', name='output')(x)

    model = Model(inputs=model_input, outputs=x)

    # Compile model
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])
    
    model.summary()
    return model

def train_eeg_model(x_train_new, x_test_new, y_train, y_test):
  
    # training the model
    model = eeg_model()
    model.fit(
        x_train_new, y_train,
        validation_data=(x_test_new, y_test),
        epochs=100)
    
    return model