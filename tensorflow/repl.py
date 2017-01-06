import tensorflow as tf
tf.__version__

tf.__path__


hello = tf.constant('Hello')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print(sess.run( a+b ))



import os
import inspect
import tensorflow
print(os.path.dirname(inspect.getfile(tensorflow)))
