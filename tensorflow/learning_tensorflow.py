import resource
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.ion()

x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y))

# tf_log_counter = 0
tf_log_counter += 1
file_writer = tf.summary.FileWriter('tf_logs/{}'.format(tf_log_counter)
        , sess.graph)


# Ex 1
x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print(session.run(y))

#
# Ex 2
x = np.random.randint(1000, size=10000)
y = tf.Variable(5*x*x - 3*x + 15)

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print(session.run(y))

#
# Ex 3
x = tf.Variable(0, name='x')

model = tf.global_variables_initializer()
with tf.Session() as session:
    for i in range(5):
        session.run(model)
        x = x + 1
        print(session.run(x))

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tf_logs/3", session.graph)
    model = tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))

#
# Lesson 3: Arrays
filename = "/tmp/MarshOrchid.jpg"
image = mpimg.imread(filename)

print(image.shape)

plt.imshow(image)
plt.show()

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    x = tf.transpose(x, perm=[1, 0, 2])
    sess.run(model)
    result = sess.run(x)

plt.imshow(result)
plt.show()

#
# Lesson 4: placeholders
x = tf.placeholder("float", 3, name='x')
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("tf_logs/0", session.graph)

filename = "/tmp/MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3], name='image')
slice = tf.strided_slice(image, [1000, 0, 0], [3000, -1, 3]
        , strides=[10, 10, 1]
        , name='sslice')
print(slice)
pad = tf.pad(slice, [[100, 100], [100, 100], [0, 0]]
        , mode='CONSTANT', name='pad0')

with tf.Session() as session:
    result = session.run(pad, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()

#
# Lession 5: Interactive Sessions
session = tf.InteractiveSession()

x = tf.constant(list(range(10)))

print(x.eval())

session.close()

print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session = tf.InteractiveSession()

X = tf.constant(np.eye(10000))
Y = tf.constant(np.random.randn(10000,300))

print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

Z = tf.matmul(X, Y)

Z.eval()

session.close()

#
# Lesson 5: TensorBoard
tf.reset_default_graph()
# Here we are defining the name of the graph, scopes A, B and C.
with tf.name_scope("MyOperationGroup"):
    with tf.name_scope("Scope_A"):
        a = tf.add(1, 2, name="Add_these_numbers")
        b = tf.mul(a, 3)
    with tf.name_scope("Scope_B"):
        c = tf.add(4, 5, name="And_These_ones")
        d = tf.mul(c, 6, name="Multiply_these_numbers")
with tf.name_scope("Scope_C"):
    e = tf.mul(4, 5, name="B_add")
    f = tf.div(c, 6, name="B_mul")
g = tf.add(b, d, name='g')
h = tf.mul(g, f, name='h')


with tf.Session() as sess:
    print(sess.run(h))

with tf.Session() as sess:
    writer = tf.summary.FileWriter("tf_logs/lesson5_scopes", sess.graph)
    print(sess.run(h))
    writer.close()

#
# Lesson 6: Reading files
filename = "olympics2016.csv"

features = tf.placeholder(tf.int32, shape=[3], name='features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')

printerop = tf.Print(total, [country, features, total], name='printer')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            country_name, code, gold, silver, bronze, total = line.strip(
                                                                ).split(",")
            gold = int(gold)
            silver = int(silver)
            bronze = int(bronze)
            # Run the Print ob
            total = sess.run(printerop,
                        feed_dict={features: [gold, silver, bronze],
                        country: country_name})
            print(country_name, total)
