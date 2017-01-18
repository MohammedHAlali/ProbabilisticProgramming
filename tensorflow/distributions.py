import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


sess = tf.InteractiveSession()


tf.reset_default_graph()

ndist = tf.contrib.distributions.Normal(0.0, 1.0)

s = ndist.sample([3])
s.name

p = ndist.pdf([0., 1.1])
p.name


p.eval()



model = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("tf_logs/dist", sess.graph)
    print(sess.run(model))
    writer.close()



tf.reset_default_graph()

ddist = tf.contrib.distributions.Bernoulli(p=[0.1, 0.5, 0.9])

s = ddist.sample(10)

s.eval()


cdist = tf.contrib.distributions.Categorical([0.2, 0.3, 0.5])

s = cdist.sample(100)

s.eval()



Dists = tf.contrib.distributions

concrete = tf.contrib.distributions.RelaxedOneHotCategorical()
