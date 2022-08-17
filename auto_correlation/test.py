import tensorflow as tf
from models import StrideDilatedNet

list_stride = [3, 2]
output_width = 1

model = StrideDilatedNet(list_stride=list_stride,
                         nb_filters=32,
                         kernel_size=2,
                         padding='causal',
                         target_size=output_width,
                         dropout_rate=0.0)

input_shape = (1, 10, 1)
x = tf.random.normal(input_shape)

y = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='causal')(x)
