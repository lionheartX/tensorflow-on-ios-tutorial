# TRAINING RUNS IN AN INFINITE LOOP, STOP IT USING Ctrl+C once converge


import os
import numpy as np
import tensorflow as tf

checkpoint_dir = "/tmp/voice/"
print_every = 1000
save_every = 10000
num_inputs = 20
num_classes = 1

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

print("Training set size:", X_train.shape)

with tf.name_scope("hyperparameters"):
    regularization = tf.placeholder(tf.float32, name="regularization")
    learning_rate = tf.placeholder(tf.float32, name="learning-rate")

# [2000, 20] means 2000 x 20 matrix/tensor
# X = 2000 x 20
# W = 20 x 1
# b = 2000 x 1, but bias is a scalar
# o = X * W + b = 2000 x 1
with tf.name_scope("inputs"):
	x = tf.placeholder(tf.float32, [None, num_inputs], name="x-input")
	y = tf.placeholder(tf.float32, [None, num_classes], name="y-input")

# weights and biases are trainable Variables
with tf.name_scope("model"):
    W = tf.Variable(tf.zeros([num_inputs, num_classes]), name="W")
    b = tf.Variable(tf.zeros([num_classes]), name="b")
    y_pred = tf.sigmoid(tf.matmul(x, W) + b, name="y_pred")

with tf.name_scope("loss-function"):
    loss = tf.losses.log_loss(labels=y, predictions=y_pred)
    #L2 regularization, prevent overfitting
    loss += regularization * tf.nn.l2_loss(W)


# Adam Optimizer uses individual momentum (adaptive learning rates) 
# vs the traditional gradient descent approach
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope("score"):
    correct_prediction = tf.equal(tf.to_float(y_pred > 0.5), y)
    # same thing as np.mean
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")

#inference for predicting non-labeled data
with tf.name_scope("inference"):
    inference = tf.to_float(y_pred > 0.5, name="inference")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.gfile.MakeDirs(checkpoint_dir)

with tf.Session() as sess:
	# write computational graph to <checkpoint_dir>/graph.pb   (trained graphs can
	# deployed directly on iOS app)
	tf.train.write_graph(sess.graph_def, checkpoint_dir, "graph.pb", False)
	sess.run(init)

	# Sanity check: the initial loss should be 0.693146, which is -ln(0.5).
	loss_value = sess.run(loss, feed_dict={x: X_train, y: y_train, regularization: 0})
	print("Initial loss:", loss_value)

	step = 0

	# this training is actually running in an infinite loop
	# TODO: stop when converges
	while True:
		# randomly shuffle the training set (permutation)
		perm = np.arange(len(X_train))
		np.random.shuffle(perm)
		X_train = X_train[perm]
		y_train = y_train[perm]

		feed = {x: X_train, y: y_train, learning_rate: 1e-2, 
                regularization: 1e-5}
		sess.run(train_op, feed_dict=feed)

		if step % print_every == 0:
			train_accuracy, loss_value = sess.run([accuracy, loss], 
                                                  feed_dict=feed)
			print("step: %4d, loss: %.4f, training accuracy: %.4f" % \
                    (step, loss_value, train_accuracy))
		step += 1
		if step % save_every == 0:
			checkpoint_file = os.path.join(checkpoint_dir, "model")
			saver.save(sess, checkpoint_file) # in /tmp/voice/
			print("*** SAVED MODEL ***")
