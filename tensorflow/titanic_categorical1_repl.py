

tf.reset_default_graph()
hot1 = tf.one_hot(y_train, 2, 1.0, 0.0)


model = tf.global_variables_initializer()
with tf.Session() as sess:
    print(sess.run(model))
    print(sess.run(hot1))

tf.contrib.layers.embed_sequence

classifier.params


for v in tf.global_variables():
    print(v.name)
