import sys

sys.path.insert(0, '../')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot as plt

# Settings:
# result_path = 'househole_result'
result_path = 'cnu_result/T100_kernal32_25-60'
# result_path = 'cnu_result/T100_kernal32'
list_stride = [24, 7]  # strides
# dilations = [24, 7]  # dilations => Khoang cach giua cac connection trong 1 filter
kernel_size = 3  # kernel_size

import tensorflow as tf

layers = tf.keras.layers

from models import StrideDilationNetDetail, StrideDilatedNet

# test:
import numpy as np


def model1_test():
    x = tf.convert_to_tensor(np.random.random((100, history_len, 1)))  # batch, seq_len, dim (number of attribute)

    model1 = StrideDilatedNet(list_stride=list_stride,
                              nb_filters=64,
                              kernel_size=kernel_size,
                              padding='causal',
                              target_size=24,
                              dropout_rate=0.0)
    y = model1(x)
    print(y.shape)
    print(model1.summary())


# model1_test()

history_len = 100
input_width = history_len
output_width = 1
num_features = 1

from utils import AreaEnergy, TSF_Data, HouseholdDataLoader

공대7호관_HV_02 = AreaEnergy('공대7호관.HV_02',
                         path_time=r"C:/Users/Andrew/Documents/Project/Time Series/Kepco-Search/dataset/Electricity data_CNU/3.unit of time(일보)/")

# dataload = HouseholdDataLoader(data_path="/home/andrew/Time Series/dataset/Household_power_consumption/household_power_consumption.txt")
# data = dataload.data_by_hour

for output_width in range(60, 100):
    orig_stdout = sys.stdout
    f = open(result_path + f'/T={history_len}-out={output_width}.txt', 'w')
    sys.stdout = f

    model = StrideDilationNetDetail(list_stride=list_stride,
                                    nb_filters=32,
                                    kernel_size=kernel_size,
                                    padding='causal',
                                    target_size=output_width,
                                    dropout_rate=0.0)

    input_test = layers.Input(shape=(input_width, num_features))

    model(input_test)
    model.summary()

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    # from tensorflow import keras
    #
    # keras.utils.plot_model(model, "Dilated on CNU.png",
    #                        dpi=120,
    #                        show_shapes=True,
    #                        show_dtype=True,
    #                        show_layer_names=True,
    #                        rankdir='TB',
    #                        expand_nested=True,
    #                        show_layer_activations=True
    #                        )

    checkpoint_path = "CNU/cp.ckpt"

    tsf = TSF_Data(data=공대7호관_HV_02.arr_seq_dataset,
                   input_width=input_width,
                   output_width=output_width,
                   train_ratio=0.9)

    # tsf = TSF_Data(data=data['Global_active_power'],
    #                input_width=input_width,
    #                output_width=output_width,
    #                train_ratio=0.9)

    tsf.normalize_data()

    print('Train...')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      mode='min')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='min')

    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    callbacks = [
        early_stopping,
        reduceLR
    ]

    history = model.fit(x=tsf.data_train[0],
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
    model.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                   verbose=2,
                   use_multiprocessing=True)

    sys.stdout = orig_stdout
    f.close()

    del model

    from matplotlib import pyplot as plt

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('DelayedNet')
    plt.savefig(result_path + "/" + f'{output_width}' + ".png", dpi=1200)
    plt.clf()
