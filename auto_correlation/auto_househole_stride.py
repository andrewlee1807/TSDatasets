import sys

sys.path.insert(0, '../')

import os
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot as plt

# Settings:
result_path = 'household_auto'

import tensorflow as tf
import keras_tuner as kt

layers = tf.keras.layers

from models import StrideDilationNetDetail, StrideDilatedNet

from utils import AreaEnergy, TSF_Data, HouseholdDataLoader

dataload = HouseholdDataLoader(
    data_path="/home/andrew/Time Series/dataset/Household_power_consumption/household_power_consumption.txt")
data = dataload.data_by_hour


def model_builder(hp):
    kernel_size = hp.Choice('kernel_size', values=[2, 3, 5, 7])
    nb_filters = hp.Choice('nb_filters', values=[8, 16, 32, 64])
    dropout_rate = hp.Float('dropout_rate', 0, 0.5, step=0.1, default=0.5)
    layer_stride1 = hp.Choice('layer_stride1', values=range(1, 24))
    layer_stride2 = hp.Choice('layer_stride2', values=range(1, 7))

    model = StrideDilatedNet(list_stride=(layer_stride1, layer_stride2),
                             nb_filters=nb_filters,
                             kernel_size=kernel_size,
                             padding='causal',
                             target_size=output_width,
                             dropout_rate=dropout_rate)

    input_test = layers.Input(shape=(input_width, num_features))
    model(input_test)
    model.summary()

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    return model


history_len = 168
input_width = history_len
num_features = 1
max_trials = 20

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  mode='min')

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
callbacks = [
    early_stopping,
    reduceLR
]

for output_width in range(1, 25):
    # Search model
    exp_path = "Household_stride_Tune/Bayesian/" + str(output_width) + "/"
    tuning_path = exp_path + "/models"

    if os.path.isdir(tuning_path):
        import shutil

        shutil.rmtree(tuning_path)

    tsf = TSF_Data(data=data['Global_active_power'],
                   input_width=input_width,
                   output_width=output_width,
                   train_ratio=0.9)

    tsf.normalize_data()

    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        seed=42,
        directory=tuning_path)

    orig_stdout = sys.stdout
    f = open(result_path + f'/T={history_len}-out={output_width}.txt', 'w')
    sys.stdout = f

    tuner.search(tsf.data_train[0], tsf.data_train[1],
                 validation_data=tsf.data_valid,
                 callbacks=[tf.keras.callbacks.TensorBoard(exp_path + "/log")],
                 epochs=10)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model_best = tuner.hypermodel.build(best_hps)

    # Train real model_searching
    print(f"""
        kernel_size {best_hps.get('kernel_size')}
        nb_filters: {best_hps.get('nb_filters')} 
        dropout_rate: {best_hps.get('dropout_rate')}
        layer_stride1: {best_hps.get('layer_stride1')}
        layer_stride2: {best_hps.get('layer_stride2')}
        """)

    print('Train...')

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

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('StrideDilatedNet after tuning')
    plt.savefig(result_path + "/" + f'{output_width}' + ".png", dpi=1200)
    plt.clf()

    del model_best
    del tuner, best_hps
