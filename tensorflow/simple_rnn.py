import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()


class SimpleRnn(object):
    def __init__(self, hidden_layer_size, input_size):
        self.hidden_layer_size = hidden_layer_size
        W_h_shape = [self.hidden_layer_size, self.hidden_layer_size]
        b_h_shape = [self.hidden_layer_size]
        W_o_shape = [self.hidden_layer_size, self.output_size]
        b_o_shape = [self.output_size]
        self.W_h = tf.get_variable('W_h', shape=W_h_shape)
        self.b_h = tf.get_variable('b_h', shape=b_h_shape)
        self.h = tf.matmul()
        self.W_o = tf.get_variable('W_o', shape=W_o_shape)
        self.b_o = tf.get_variable('b_o', shape=b_o_shape)
        self.o = tf.nn.xw_plus_b(states, W_o, b_o)

        self.loss = tf.reduce_mean((self.targets - self.o)**2, name='loss')



