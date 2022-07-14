import sys

sys.path.insert(0, r'C:\Users\Andrew\Documents\Project\Time Series\TSDatasets')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Settings:
result_path = 'househole_result'
# result_path = 'cnu_result'
history_len = 48
list_stride = [24, 7]  # strides
# dilations = [24, 7]  # dilations => Khoang cach giua cac connection trong 1 filter
kernel_size = 3  # kernel_size

import tensorflow as tf

layers = tf.keras.layers


class DilatedLayer(layers.Layer):
    def __init__(self,
                 nb_stride=3,
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 dropout_rate=0.0,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="DilatedLayer", **kwargs):
        super(DilatedLayer, self).__init__(name=name, **kwargs)

        self.conv1 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=nb_stride,
                                   padding=padding,
                                   name='conv1D',
                                   kernel_initializer=init)

        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x)
        return x


class DelayedNet(tf.keras.Model):
    def __init__(self,
                 list_stride=(3, 3),
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0):
        self.nb_filters = nb_filters
        self.list_stride = list_stride
        self.kernel_size = kernel_size

        super(DelayedNet, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        self.dilation1 = DilatedLayer(nb_stride=list_stride[0],
                                      nb_filters=nb_filters,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      init=init,
                                      dropout_rate=dropout_rate,
                                      name='DilatedLayer_1')

        self.dilation2 = DilatedLayer(nb_stride=list_stride[1],
                                      nb_filters=nb_filters,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      init=init,
                                      dropout_rate=dropout_rate,
                                      name='DilatedLayer_2')

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

        # add this code
        self.call(layers.Input(shape=(history_len, 1)))

    def call(self, inputs, training=True):
        x = self.dilation1(inputs)
        x = self.dilation2(x)
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x

    @property
    def receptive_field(self):
        return history_len * np.prod(self.list_stride) - np.prod(self.list_stride) - \
               self.list_stride[0] * (1 - self.kernel_size) + self.kernel_size


class DelayedNetDetail(tf.keras.Model):
    def __init__(self,
                 list_stride=(3, 3),
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0):
        self.nb_filters = nb_filters

        super(DelayedNetDetail, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        # D0
        self.conv1 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=list_stride[0],
                                   padding=padding,
                                   name='conv1D_1',
                                   kernel_initializer=init)

        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

        # D1
        self.conv2 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=list_stride[1],
                                   padding=padding,
                                   name='conv1D_2',
                                   kernel_initializer=init)

        self.batch2 = layers.BatchNormalization(axis=-1)
        self.ac2 = layers.Activation('relu')
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

        # add this code
        self.call(layers.Input(shape=(history_len, 1)))

    def call(self, inputs, training=True):
        prev_x = inputs
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        # if prev_x.shape[-1] != x.shape[-1]:  # match the dimention
        #     prev_x = self.downsample(prev_x)
        # assert prev_x.shape == x.shape

        # return self.ac3(prev_x + x)  # skip connection
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x

    @property
    def receptive_field(self):
        return history_len * np.prod(self.list_stride) - np.prod(self.list_stride) - \
               self.list_stride[0] * (1 - self.kernel_size) + self.kernel_size


# test:
import numpy as np


def model1_test():
    x = tf.convert_to_tensor(np.random.random((100, history_len, 1)))  # batch, seq_len, dim (number of attribute)

    model1 = DelayedNet(list_stride=list_stride,
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

for output_width in range(1, 25):
    orig_stdout = sys.stdout
    f = open(result_path + f'/T={history_len}-out={output_width}.txt', 'w')
    # f = open(result_path + f'/seaching_process_log_cnu_T={str(history_len)}.txt', 'w')
    sys.stdout = f

    model = DelayedNetDetail(list_stride=(24, 7),
                             nb_filters=128,
                             kernel_size=3,
                             padding='causal',
                             target_size=24,
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

    from utils import AreaEnergy, TSF_Data, HouseholdDataLoader

    # 공대7호관_HV_02 = AreaEnergy('공대7호관.HV_02',
    #                          path_time=r"C:/Users/Andrew/Documents/Project/Time Series/Kepco-Search/dataset/Electricity data_CNU/3.unit of time(일보)/")
    #
    # tsf = TSF_Data(data=공대7호관_HV_02.arr_seq_dataset,
    #                input_width=input_width,
    #                output_width=output_width,
    #                train_ratio=0.9)

    dataload = HouseholdDataLoader(data_path="C:/Users/Andrew/Documents/Project/Time Series/Kepco-Search/dataset/Household_power_consumption/household_power_consumption.txt")
    data = dataload.data_by_hour

    tsf = TSF_Data(data=data['Global_active_power'],
                   input_width=input_width,
                   output_width=output_width,
                   train_ratio=0.9)

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

    from matplotlib import pyplot as plt

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('DelayedNet')
    plt.savefig(result_path + "/" + f'/T={history_len}-out={output_width}' + ".png", dpi=1200)
    plt.clf()
