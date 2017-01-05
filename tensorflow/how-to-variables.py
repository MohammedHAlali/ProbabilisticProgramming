import tensorflow as tf

weights = tf.Variable( 
            tf.random_normal( [784,200] , stddev=0.35 )
            , name="weights" )
biases = tf.Variable(
            tf.zeros([200])
            , name="biases" )

print(biases)

with tf.device( "/cpu:0" ):
    b = tf.Variable( tf.zeros([20]) , name="b1" )

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

ckpt_file = "/tmp/ckpt/model.checkpoint" 

with tf.Session() as sess:
    sess.run( init_op )
    save_path = saver.save( sess , ckpt_file )
    print( "Model saved in: %s" % save_path )

with tf.Session() as sess:
    saver.restore( sess , "/tmp/model.checkpoint" )
