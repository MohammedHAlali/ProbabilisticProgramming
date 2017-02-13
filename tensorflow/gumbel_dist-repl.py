import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

rng = np.random.RandomState(42)


def gumbel(*args):
    u = rng.rand(*args)
    return -np.log(-np.log(u))


dat = gumbel(10000)
plt.clf()
plt.hist(dat, 50)



tf.reset_default_graph()
sess = tf.InteractiveSession()

batch_size = 3
number_of_classes = 5

init = tf.constant(np.linspace(-3, 3, number_of_classes, dtype=np.float32))
logits = tf.get_variable('logits', 
        initializer=init)

init_logits = tf.variables_initializer([logits])
sess.run(init_logits)
logits.eval()
logits.name
logits.dtype

uniform = tf.random_uniform((number_of_classes,), 0,1)
uniform.dtype

eps = 1e-20
gumbel = -tf.log(-tf.log(uniform + eps) + eps)

y_soft = tf.nn.softmax(logits+gumbel, name='y_soft')
max = tf.reduce_max(y_soft, 0, keep_dims=True)
eq = tf.equal(y_soft, max)
y_hard = tf.cast(eq, y_soft.dtype, name='y_hard')
sg = tf.stop_gradient(y_hard - y_soft)
y_g = tf.add(sg, y_soft, name='y_g')
print('y_soft: ', y_soft.eval())
print('max: ', max.eval())
print('eq: ', eq.eval())
print("y_hard: ", y_hard.eval())
print("sg: ", sg.eval())
print('y_g', y_g.eval())

argmax_y = tf.arg_max(y_soft, 0)
y_hard = tf.one_hot(argmax_y, number_of_classes)
print('argmax: ', argmax_y.eval())
print('onehot: ', y_hard.eval())
sg = tf.stop_gradient(y_hard - y_soft)
y_g = tf.add(sg, y_soft, name='y_g')

grad = tf.gradients(y_g, [logits])
grad[0].eval()

input_data = tf.placeholder_with_default(
        np.array([2., 8., 1., 3., 2.], dtype=np.float32),
        number_of_classes,
        name='input_data')
loss = tf.reduce_sum(tf.mul(input_data, y_g), name='loss')
loss.eval()

train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(
        loss,
        var_list=[logits])

NUM_ITER = 400
model = tf.global_variables_initializer()
sess.run(model)
losses = []
theta = np.zeros((NUM_ITER,number_of_classes))
for i in range(NUM_ITER):
    _, np_loss, np_logits = sess.run([train_op, loss, y_g])
    losses.append(np_loss)
    theta[i] = np_logits
    print(np_logits)

fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
for i in range(number_of_classes):
    ax.plot( range(NUM_ITER), theta[:,i], label='class {}'.format(i))
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()

plt.matshow(theta, aspect=0.03, fignum=2, cmap='hot')


writer = tf.summary.FileWriter("tf_logs/3", sess.graph)
writer.close()


#
# as module

%load_ext autoreload
%autoreload 2

from gumbel_dist import Concrete

tf.reset_default_graph()
sess = tf.InteractiveSession()
cat = Concrete(5)
s = cat.sample()

init_op = tf.global_variables_initializer()
sess.run(init_op)

cat.logits.eval()

s.eval()

np.random.randn(4)

