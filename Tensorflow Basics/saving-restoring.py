import tensorflow as tf

saver = tf.train.Saver()

with tf.Session() as sess:
    .
    .
    .
    .
    saver.save(sess,'models/mymodel.ckpt')

with tf.Session() as sess:
    saver.restore(sess,'models/mymodel.ckpt')
