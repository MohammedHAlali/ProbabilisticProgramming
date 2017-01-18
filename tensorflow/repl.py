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


tf.reset_default_graph()


model = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("tf_logs/", sess.graph)
    sess.run(model)
    writer.close()

for v in tf.global_variables():
    print(v.name)

for v in tf.trainable_variables():
    print(v.name)
    print(sess.run(v))

for op in sess.graph.get_operations():
    print(op.name)
