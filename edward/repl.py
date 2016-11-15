import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal, Bernoulli, Beta
import edward as ed
plt.ion()

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


#
#
#

# 1. Preloaded data
x_data = np.array( [0, 1, 0, 0, 0, 0, 0, 0, 0, 1] )

x_data_tf = tf.constant( x_data )

# 2. Feeding 
x_data = tf.placeholder( tf.float32, [100,25] )

# 3. Reading from files
filename_queue = tf.train.string_input_producer( )


# model
from edward.models import Normal

x = Normal(mu=tf.zeros(10), sigma=tf.ones(10))
y = tf.constant(5.0)
x + y, x - y, x * y, x / y
tf.tanh(x * y)
tf.gather(x, 2)  # 3rd normal rv in the vector

#
# Composing random variables
theta = Beta( a=1.0, b=1.0 )

x = Bernoulli( p=tf.ones(50) * theta )

theta = tf.Variable( 0.0 )

x = Bernoulli( p=tf.ones( 50 ) * tf.sigmoid( theta ) )

#
# Neural nets
from edward.models import Bernoulli, Normal
from tensorflow.contrib import slim

z = Normal( mu=tf.zeros( [N,d] ), sigma=tf.ones( [N,d] ) )

#
# Bayesian Non-parametrics
# 1. Collapsing the infinite-dim space
#    - Integrate out gaussian process
# 2. Lazily defining the infinite-dim space
#    - control flow operations in TF

#
# Criticism
# 1. point-based: MSE / classification accuracy 
# 2. posterior-predictive-checks
from edward.models import Categorical, Normal

# MODEL
K = 3
D = 2
N = 4
beta = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
z = Categorical(logits=tf.zeros([N, K]))
x = Normal(mu=tf.gather(beta, z), sigma=tf.ones([N, D]))

# INFERENCE
qbeta = Normal(mu=tf.Variable(tf.zeros([K, D])),
               sigma=tf.exp(tf.Variable(tf.zeros([K, D]))))
qz = Categorical(logits=tf.Variable(tf.zeros([N, K])))

x_train = 

inference = ed.Inference({z: qz, beta: qbeta}, data={x: x_train})
inference.run()

#
# Supervised learning (regression)
def build_toy_dataset(N, coeff=np.random.randn(10), noise_std=0.1):
  n_dim = len(coeff)
  x = np.random.randn(N, n_dim).astype(np.float32)
  y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=N)
  return x, y

N = 40  # number of data points
D = 10  # number of features

coeff = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, coeff)
X_test, y_test = build_toy_dataset(N, coeff)

plt.plot(X_train, y_train )


fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.scatter( X_train[:,1], y_train , label=''  , color='blue' )
ax.scatter( X_train[:,9], y_train , label=''  , color='red' )
ax.scatter( X_train[:,8], y_train , label=''  , color='green' )
ax.scatter( X_train[:,6], y_train , label=''  , color='orange' )
fig.canvas.draw()
fig.show()


X = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

data = {X: X_train, y: y_train}
inference = ed.KLqp({w: qw, b: qb}, data)
inference.run()
