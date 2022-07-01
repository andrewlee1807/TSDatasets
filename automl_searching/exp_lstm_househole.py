import sys
sys.path.insert(0, '/home/dspserver/andrew/TSDatasets')
from utils import HouseholdDataLoader, TSF_Data

dataload = HouseholdDataLoader()
data = dataload.data_by_days

import os
import pandas as pd


from sklearn.model_selection import train_test_split
import numpy as np
from utils import TSF_Data

tsf = TSF_Data(data=data['Global_active_power'],
               input_width=21,
               output_width=7,
               train_ratio=0.8,
               shuffle=True)
tsf.normalize_data(standardization_type=1)


for output_width in range(1, 25):
    # Search model
    exp_path = "HouseHold_TCN_Tune/Bayesian/"+str(output_width)+"/"
    tuning_path = exp_path + "/models"

    if os.path.isdir(tuning_path):
        import shutil
        shutil.rmtree(tuning_path)

        
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import Sequential
from keras.layers import Dense, LSTM

model_tsf = Sequential()
model_tsf.add(LSTM(200, return_sequences=True, activation='relu',
              input_shape=(tsf.data_train[0].shape[1], 1)))
model_tsf.add(LSTM(150))
model_tsf.add(Dense(7))

print(model_tsf.summary())
model_tsf.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
]

history_tsf = model_tsf.fit(x=tsf.data_train[0],
                            y=tsf.data_train[1],
                            epochs=100, validation_data=tsf.data_valid,
                            batch_size=32,
                            steps_per_epoch=100,
                            callbacks=callbacks)
