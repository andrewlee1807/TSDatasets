import sys
sys.path.insert(0, '/home/dspserver/andrew/TSDatasets')
from utils import HouseholdDataLoader, TSF_Data

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataload = HouseholdDataLoader()
data = dataload.data_by_days

result_path = "household_result_lstm" # saving the processing of training phase and images ploted

import pandas as pd


from sklearn.model_selection import train_test_split
import numpy as np
from utils import TSF_Data

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import Sequential
from keras.layers import Dense, LSTM

def build_model(tsf, output_width):
    model_tsf = Sequential()
    model_tsf.add(LSTM(200, return_sequences=True, activation='relu',
                input_shape=(tsf.data_train[0].shape[1], 1)))
    model_tsf.add(LSTM(150))
    model_tsf.add(Dense(output_width))

    print(model_tsf.summary())
    model_tsf.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model_tsf


input_width = 24
callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
]


for output_width in range(1, 25):
    # Search model
    tsf = TSF_Data(data=data['Global_active_power'],
               input_width=input_width,
               output_width=output_width,
               train_ratio=0.9)
    tsf.normalize_data(standardization_type=1)

    model_tsf= build_model(tsf)

    orig_stdout = sys.stdout
    f = open(result_path + f'/seaching_process_log_cnu_{str(output_width)}.txt', 'w')
    sys.stdout = f

    history = model_tsf.fit(x=tsf.data_train[0],
                            y=tsf.data_train[1],
                            epochs=100, validation_data=tsf.data_valid,
                            batch_size=32,
                            steps_per_epoch=100,
                            callbacks=callbacks)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    model_tsf.evaluate(tsf.data_test[0],tsf.data_test[1], batch_size=1,
               verbose=2,
               use_multiprocessing=True)
    sys.stdout = orig_stdout
    f.close()

    del model_tsf, tsf

    from matplotlib import pyplot as plt
    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('TCN after tunning')
    # plt.show()
    plt.savefig(result_path + "/" + str(output_width) + ".png", dpi=1200)
    plt.clf()