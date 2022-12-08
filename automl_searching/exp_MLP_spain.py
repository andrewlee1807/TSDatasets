import sys

sys.path.insert(0, '../')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

result_path = "spain/spain_result_mlp" # saving the processing of training phase and images ploted
num_features = 1
input_width = 168

from matplotlib import pyplot as plt


from utils import TSF_Data

import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ultils_mlp import *


class MLP_BASE(tf.keras.Model):
    def __init__(self,
                 input_width=168,
                 target_size=24,
                 num_features=1,
                 **kwargs):
        super(MLP_BASE, self).__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(units=128, input_dim=input_width, activation='relu')
        self.layer2 = layers.Dense(8, activation='relu')
        self.dense = layers.Dense(target_size, activation='linear')

    def call(self, inputs):
        x = inputs
        # flatten input
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dense(x)
        return x


def build_model(input_width, output_width):
    model = MLP_BASE(input_width=input_width,
                    target_size=output_width,
                    num_features=1)
    
    input_test = layers.Input(shape=(input_width, num_features))
    model(input_test)
    model.summary()
    
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])


    return model


callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
]

# Get dataset
dataset = Dataset(dataset_name="SPAIN")
raw_data = dataset.dataloader.export_sequences()

for output_width in [1, 12, 24, 36, 48, 60, 72, 84]:
    # Search model
    tsf = TSF_Data(data=raw_data,
                input_width=input_width,
                output_width=output_width,
                train_ratio=0.9)

    tsf.normalize_data()

    orig_stdout = sys.stdout
    f = open(result_path + f'/seaching_process_log_{str(output_width)}.txt', 'w')
    sys.stdout = f

    model_tsf= build_model(input_width, output_width)

    history = model_tsf.fit(x=tsf.data_train[0],
                            y=tsf.data_train[1],
                            epochs=100, 
                            validation_data=tsf.data_valid,
                            batch_size=32,
                            verbose=2,
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

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('MLP')
    # plt.show()
    plt.savefig(result_path + "/" + str(output_width) + ".png", dpi=120)
    plt.clf()
