import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(a + b))