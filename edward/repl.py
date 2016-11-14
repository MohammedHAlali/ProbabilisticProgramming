import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal
import edward as ed

x_train = np.linspace( -3, 3, num=50 )

y_train = np.cos( x_train ) + norm.rvs( 0, 0.1, size=50 )

plt.plot(x_train, y_train, 'ro')
plt.show()


W_0 = Normal(mu=tf.zeros([1, 2]), sigma=tf.ones([1, 2]))
W_1 = Normal(mu=tf.zeros([2, 1]), sigma=tf.ones([2, 1]))
b_0 = Normal(mu=tf.zeros(2), sigma=tf.ones(2))
b_1 = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

x = tf.convert_to_tensor(x_train, dtype=tf.float32)

# ValueError: Shape must be rank 2 but is rank 1
tf.matmul(x, W_0)

y = Normal(mu=tf.matmul(tf.tanh(tf.matmul(x, W_0) + b_0), W_1) + b_1
        , sigma=0.1)

y = Normal(mu=tf.matmul(tf.tanh(tf.matmul(x, W_0) + b_0), W_1) + b_1)

qW_0 = Normal(mu=tf.Variable(tf.random_normal([1, 2])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1, 2]))))
qW_1 = Normal(mu=tf.Variable(tf.random_normal([2, 1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([2, 1]))))
qb_0 = Normal(mu=tf.Variable(tf.random_normal([2])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([2]))))
qb_1 = Normal(mu=tf.Variable(tf.random_normal([1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


data = {y: y_train}
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data)
inference.run(n_iter=1000)
