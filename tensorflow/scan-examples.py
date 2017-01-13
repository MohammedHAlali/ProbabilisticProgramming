# https://rdipietro.github.io/tensorflow-scan-examples/

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
plt.ion()


def fn(previous_output, current_input):
    return previous_output + current_input


tf.reset_default_graph()
elems = tf.Variable([1.0, 2.0, 2.0, 2.0], name='var_elems')
# elems = tf.identity(elems)
initializer = tf.constant(0.0)
out = tf.scan(fn, elems, initializer=initializer)

model = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("tf_logs/scan-ex", sess.graph)
    sess.run(model)
    print(sess.run(out))
    writer.close()


def input_target_generator(min_duration=5, max_duration=50):
    """ Generate toy input, target sequences.

    Each input sequence has values that are drawn from the standard normal
    distribution, and each target sequence is the corresponding cumulative sum.
    Sequence durations are chosen at random using a discrete uniform
    distribution over `[min_duration, max_duration]`.

    Args:
        min_duration: A positive integer. The minimum sequence duration.
        max_duration: A positive integer. The maximum sequence duration.

    Yields:
        A tuple,
        inputs: A 2-D float32 NumPy array with shape `[duration, 1]`.
        targets: A 2-D float32 NumPy array with shape `[duration, 1]`.
    """

    while True:
        duration = np.random.randint(min_duration, max_duration)
        inputs = np.random.randn(duration).astype(np.float32)
        targets = np.cumsum(inputs).astype(np.float32)
        yield inputs.reshape(-1, 1), targets.reshape(-1, 1)


class Model(object):
    def __init__(self, hidden_layer_size, input_size, target_size,
                 init_scale=0.1):
        """ Create a vanilla RNN.
        Args:
            hidden_layer_size: An integer. The number of hidden units.
            input_size: An integer. The number of inputs per time step.
            target_size: An integer. The number of targets per time step.
            init_scale: A float. All weight matrices will be initialized using
                a uniform distribution over [-init_scale, init_scale].
        """
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.target_size = target_size
        self.init_scale = init_scale
        self._inputs = tf.placeholder(tf.float32, shape=[None, input_size],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32, shape=[None, target_size],
                                       name='targets')
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._states, self._predictions = self._compute_predictions()
            self._loss = self._compute_loss()

    def _vanilla_rnn_step(self, h_prev, x):
        """ Vanilla RNN step.

        Args:
            h_prev: A 1-D float32 Tensor with shape `[hidden_layer_size]`.
            x: A 1-D float32 Tensor with shape `[input_size]`.

        Returns:
            The updated state `h`, with the same shape as `h_prev`.
        """

        h_prev = tf.reshape(h_prev, [1, self.hidden_layer_size])
        x = tf.reshape(x, [1, self.input_size])

        with tf.variable_scope('rnn_block'):
            W_h = tf.get_variable(
                'W_h', shape=[self.hidden_layer_size, self.hidden_layer_size])
            W_x = tf.get_variable(
                'W_x', shape=[self.input_size, self.hidden_layer_size])
            b = tf.get_variable('b', shape=[self.hidden_layer_size],
                                initializer=tf.constant_initializer(0.0))
            h = tf.matmul(h_prev, W_h) + tf.matmul(x, W_x) + b
            # h = tf.nn.relu(h)
            h = tf.reshape(h, [self.hidden_layer_size], name='h')
        return h

    def _compute_predictions(self):
        """ Compute vanilla-RNN states and predictions. """

        with tf.variable_scope('states'):
            initial_state = tf.zeros([self.hidden_layer_size],
                                     name='initial_state')
            states = tf.scan(self._vanilla_rnn_step, self.inputs,
                             initializer=initial_state, name='states')

        with tf.variable_scope('predictions'):
            W_pred = tf.get_variable(
                'W_pred', shape=[self.hidden_layer_size, self.target_size])
            b_pred = tf.get_variable('b_pred', shape=[self.target_size],
                                     initializer=tf.constant_initializer(0.0))
            predictions = tf.add(
                    tf.matmul(states, W_pred),
                    b_pred, name='predictions')
        return states, predictions

    def _compute_loss(self):
        """ Compute l2 loss between targets and predictions. """

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(
                    (self.targets - self.predictions)**2,
                    name='loss')
            return loss

    @property
    def inputs(self):
        """ A 2-D float32 placeholder with shape
        `[dynamic_duration, input_size]`. """
        return self._inputs

    @property
    def targets(self):
        """ A 2-D float32 placeholder with shape
        `[dynamic_duration, target_size]`. """
        return self._targets

    @property
    def states(self):
        """ A 2-D float32 Tensor with shape
        `[dynamic_duration, hidden_layer_size]`. """
        return self._states

    @property
    def predictions(self):
        """ A 2-D float32 Tensor with shape
        `[dynamic_duration, target_size]`. """
        return self._predictions

    @property
    def loss(self):
        """ A 0-D float32 Tensor. """
        return self._loss


class Optimizer(object):
    def __init__(self, loss, initial_learning_rate, num_steps_per_decay,
                 decay_rate, max_global_norm=1.0):
        """ Create a simple optimizer.

        This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.

        Args:
            loss: A 0-D float32 Tensor.
            initial_learning_rate: A float.
            num_steps_per_decay: An integer.
            decay_rate: A float. The factor applied to the learning rate
                every `num_steps_per_decay` steps.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm because `max_global_norm`.
        """

        trainables = tf.trainable_variables()
        grads = tf.gradients(loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, num_steps_per_decay,
            decay_rate, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._optimize_op = optimizer.apply_gradients(grad_var_pairs,
                                                      global_step=global_step)

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op


def train(sess, model, optimizer, generator, num_optimization_steps,
          logdir='./tf_logs'):
    """ Train.

    Args:
        sess: A Session.
        model: A Model.
        optimizer: An Optimizer.
        generator: A generator that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """

    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    tf.summary.scalar('loss', model.loss)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss_ema = ema.apply([model.loss])
    loss_ema = ema.average(model.loss)
    tf.summary.scalar('loss_ema', loss_ema)

    summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in xrange(num_optimization_steps):
        inputs, targets = generator.next()
        loss_ema_, summary, _, _ = sess.run(
            [loss_ema, summary_op, optimizer.optimize_op, update_loss_ema],
            {model.inputs: inputs, model.targets: targets})
        # summary_writer.add_summary(summary, global_step=step)
        print('\rStep %d. Loss EMA: %.6f.' % (step+1, loss_ema_), end='')


tf.reset_default_graph()
generator = input_target_generator()
model = Model(
        hidden_layer_size=1, input_size=1, target_size=1, init_scale=0.1)
optimizer = Optimizer(
        model.loss, initial_learning_rate=1e-2, num_steps_per_decay=15000,
        decay_rate=0.1, max_global_norm=1.0)

import timeit
start = timeit.default_timer()
sess = tf.Session()
train(sess, model, optimizer, generator, num_optimization_steps=15000)
end = timeit.default_timer()
print((end - start), "sec")
# 66.6315 sec  w/summaries
# 60.0650 sec  w/o summaries


def test_qualitatively(
        sess, model, generator, num_examples=5, figsize=(10, 3)):
    """ Test qualitatively.

    Args:
        sess: A Session.
        model: A Model.
        generator: A generator that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_examples: An integer. The number of examples to plot.
        figsize: A tuple `(width, height)`, the size of each example's figure.
    """

    for i in xrange(num_examples):

        inputs, targets = generator.next()
        predictions = sess.run(model.predictions, {model.inputs: inputs})

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        ax[0].plot(inputs.flatten(), label='inputs')
        ax[0].legend()
        ax[1].plot(targets.flatten(), label='targets')
        ax[1].plot(predictions.flatten(), 'o', label='predictions')
        ax[1].legend()


test_qualitatively(sess, model, generator, figsize=(8, 2))


for v in tf.global_variables():
    print(v.name)

for v in tf.trainable_variables():
    print(v.name)
    print(sess.run(v))

for op in sess.graph.get_operations():
    print(op.name)

var = sess.graph.get_tensor_by_name('model/predictions/W_pred:0')
sess.run(var)

tf.get_variable_scope()

tf.verify_tensor_all_finite()

tf.add_check_numerics_ops()

fc1 = tf.nn.xw_plus_b(x, W, b)
fc1 = tf.nn.relu(fc1)

plt.matshow(sess.run(Wh))

for summary in tf.train.summary_iterator('./tf_logs'):
    print(summary)
