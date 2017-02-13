# <script src="https://gist.github.com/gngdb/ef1999ce3a8e0c5cc2ed35f488e19748.js"></script>
# http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
import tensorflow as tf
import numpy as np


class Concrete(object):
    def __init__(self, number_of_classes, temperature=1.0, seed=None):
        self.number_of_classes = number_of_classes
        self.shape = (number_of_classes,)
        with tf.variable_scope("Concrete"):
            self.logits = tf.get_variable('logits', self.shape)
            self.temperature = tf.identity(temperature, name="temperature")
        self.dtype = tf.float32
        self.seed = seed

    def sample_gumbel(self):
        """Sample from Gumbel(0, 1)"""
        with tf.name_scope('gumbel'):
            np_dtype = self.dtype.as_numpy_dtype
            minval = np.nextafter(np_dtype(0), np_dtype(1))
            uniform = tf.random_uniform(
                    shape=self.shape,
                    minval=minval,
                    maxval=1,
                    dtype=self.dtype,
                    seed=self.seed)
            gumbel = - tf.log(- tf.log(uniform))
        return gumbel

    def sample(self):
        with tf.name_scope('Concrete'):
            gumbel = self.sample_gumbel()
            noisy_logits = tf.div(gumbel + self.logits, self.temperature)
            soft_onehot = tf.nn.softmax(noisy_logits)
            argmax = tf.arg_max(soft_onehot, 0)
            hard_onehot = tf.one_hot(argmax, self.number_of_classes)
            stop_grad = tf.stop_gradient(hard_onehot - soft_onehot)
            # h = h - s + s
            differentiable_hard_onehot = tf.add(stop_grad, soft_onehot,
                    name='onehot')
        return differentiable_hard_onehot
