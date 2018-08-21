import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("World")
with tf.Session() as sess:
    result = sess.run(hello + world)
print(result)

a = tf.constant(10)
b = tf.constant(20)
with tf.Session() as sess:
    result = sess.run(a + b)
print(result)

const = tf.constant(10)
fill_mat = tf.fill((4, 4), 10)
myzeros = tf.zeros((4, 4))
myones = tf.ones((4, 4))
myrandn = tf.random_normal((4, 4), mean=0, stddev=1.0)
myrandu = tf.random_uniform((4, 4), minval=0, maxval=1)
my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]
sess = tf.InteractiveSession()
for op in my_ops:
    print(sess.run(op))
    print('\n')

a = tf.constant([[1, 2], [3, 4]])
a.get_shape()
b = tf.constant([[10], [100]])
b.get_shape()
result = tf.matmul(a, b)
print(sess.run(result))
