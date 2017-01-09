import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable( [1.,2.] )
a = tf.constant( [3.,3.] )

x.initializer.run()

sub = tf.sub( x , a )

print( sub.eval() )

sess.close()


sub = tf.sub( x, tf.constant( [0.,0.] ) )
print( sub.eval() )
