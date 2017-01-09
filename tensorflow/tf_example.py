import tensorflow as tf
import numpy as np

# Make 100 phony data points in NumPy.
x_data = np.float32(np.random.rand(2, 100)) # Random input
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# Construct a linear model.
b = tf.Variable(tf.zeros([1]),name='b')
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0),name='W')
y = tf.matmul(W, x_data) + b

# Minimize the squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# For initializing the variables.
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the plane.
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

W_fit = sess.run(W)
b_fit = sess.run(b)

# Learns best fit is W: [[0.100  0.200]], b: [0.300]

file_writer = tf.summary.FileWriter( 'feed.log' , sess.graph)

sess.close()

import matplotlib.pyplot as plt
plt.ion()

L = 10
x0 = np.linspace( -0.1, 1.1, L)
x1 = np.linspace( -0.1, 1.1, L)
x5 = np.repeat( 0.5, L )
y0 = np.dot( W_fit, np.array(zip( x0, x5 ) ).T )[0] + b_fit
y1 = np.dot( W_fit, np.array(zip( x5, x1 ) ).T )[0] + b_fit

plt.clf()
plt.subplot(3,1,1)
plt.scatter( x_data[0,:], y_data)
plt.plot( x0, y0 )
plt.xlabel('x_0')
plt.ylabel('y')
plt.subplot(3,1,2)
plt.scatter( x_data[1,:], y_data)
plt.plot( x1, y1 )
plt.xlabel('x_1')
plt.ylabel('y')
plt.subplot(3,1,3)
plt.scatter( x_data[0,:], x_data[1,:] )
plt.scatter( x5, x5, color='red', s=300, marker='x' )
plt.xlabel('x_0')
plt.ylabel('x_1')

