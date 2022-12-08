import sys

sys.path.insert(0, '../')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result_path = "cnu/cnu_result_lstm" # saving the processing of training phase and images ploted
num_features = 1
input_width = 168

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from utils import TSF_Data


import keras_tuner as kt
import os
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, LSTM
from keras import Sequential
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from ultils_mlp import *



def model_builder(hp):
    node_layer1 = hp.Choice('node_layer1', values=[30, 40, 50, 100])
    node_layer2 = hp.Choice('node_layer2', values=[30, 40, 50, 100])
    
    inputs = Input(shape=(input_width, num_features))
    x1 = LSTM(node_layer1, return_sequences=True, activation='relu',
                input_shape=(input_width, 1))(inputs)
    x2 = LSTM(node_layer2)(x1)
    x3 = Dense(output_width)(x2)
    
    model_tsf = Model(inputs, x3)

    model_tsf.summary()

    model_tsf.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    return model_tsf


max_trials = 5
# Get dataset
dataset = Dataset(dataset_name="CNU")
raw_data = dataset.dataloader.raw_data


# for output_width in range(36, 73, 12):
for output_width in [1, 12, 24, 36, 48, 60, 72, 84]:
    # Search model
    exp_path = "CNU_LSTM_Tune/Bayesian/" + str(output_width) + "/"
    tuning_path = exp_path + "/models"

    if os.path.isdir(tuning_path):
        import shutil

        shutil.rmtree(tuning_path)

    # Search model
    tsf = TSF_Data(data=raw_data,
                input_width=input_width,
                output_width=output_width,
                train_ratio=0.9)

    tsf.normalize_data()

    input_width = tsf.data_train[0].shape[1]

    inputs = Input(shape=(input_width, num_features))

    print("[INFO] instantiating a random search tuner object...")

    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        seed=42,
        directory=tuning_path)

    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    orig_stdout = sys.stdout
    f = open(result_path + f'/seaching_process_log_cnu_{str(output_width)}.txt', 'w')
    sys.stdout = f

    tuner.search(tsf.data_train[0], tsf.data_train[1],
                 validation_data=tsf.data_valid,
                 callbacks=[tf.keras.callbacks.TensorBoard(exp_path + "/log")],
                 epochs=10)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Train real model_searching

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model_best = tuner.hypermodel.build(best_hps)

    # Train real model_searching
    print(f"""
    node_layer1 {best_hps.get('node_layer1')},  and
    node_layer2: {best_hps.get('node_layer2')}
    """)

    print('Train...')

    callbacks = [
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    ]

    history = model_best.fit(x=tsf.data_train[0],
                             y=tsf.data_train[1],
                             validation_data=tsf.data_valid,
                             epochs=100,
                             callbacks=[callbacks],
                             verbose=2,
                             use_multiprocessing=True)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    model_best.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                        verbose=2,
                        use_multiprocessing=True)
    sys.stdout = orig_stdout
    f.close()

    pd.DataFrame.from_dict(history.history).to_csv(result_path + '/history' + str(output_width) + '.csv', index=False)

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

    del model_best
    del tuner, best_hps
