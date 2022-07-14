import sys
sys.path.insert(0, '/home/dspserver/andrew/TSDatasets')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from utils import AreaEnergy, TSF_Data

공대7호관_HV_02 = AreaEnergy('공대7호관.HV_02',
                         path_time=r"/home/dspserver/andrew/dataset/Electricity data_CNU/3.unit of time(일보)/")

result_patth = 'cnu_result'

import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, LSTM
from keras import Sequential
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tcn import TCN

x1 = TCN(input_shape=(input_width, 1),
        kernel_size=kernel_size,
        nb_filters=nb_filters,
        dilations=dilation_gen(dilations),
        use_skip_connections=use_skip_connections,
        use_batch_norm=use_batch_norm,
        use_weight_norm=False,
        use_layer_norm=False,
        return_sequences=False
        )(inputs)

x2 = LSTM(nb_units_lstm)(x1)

x3 = Dense(units=tsf.data_train[1].shape[1])(x1)

model_searching = Model(inputs, x3)

# Tune the learning rate for the optimizer
# Choose an optimal value from 0.01, 0.001, or 0.0001
# hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

model_searching.summary()

model_searching.compile(loss=tf.keras.losses.Huber(),
                        optimizer='adam',
                        metrics=['mse', 'mae'])
