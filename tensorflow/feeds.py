import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    data = { input1:[7.]
            , input2:[2.]}
    result = sess.run([output], feed_dict=data)
    print(result)

sess = tf.InteractiveSession()

output.eval( feed_dict=data )

sess.close()


