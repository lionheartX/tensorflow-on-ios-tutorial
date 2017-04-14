import os
import numpy as np
import tensorflow as tf
from sklearn import metrics

checkpoint_dir = "/tmp/voice/"

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Test set size:", X_test.shape)

# To compute on test set, if you have trained the model, you should have
# a graph.pb file, we're going to use this file to load the computational graph
# instead of re-writing it.

with tf.Session() as sess:
	graph_file = os.path.join(checkpoint_dir, "graph.pb")
	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name="")


	#print(graph_def.ListFields()) #print what the graph looks like

	# You can get_tensor_by_name if you give your nodes explicit names.
	W = sess.graph.get_tensor_by_name("model/W:0")
	b = sess.graph.get_tensor_by_name("model/b:0")

	checkpoint_file = os.path.join(checkpoint_dir, "model")
	saver = tf.train.Saver([W, b])
	saver.restore(sess, checkpoint_file)

	x = sess.graph.get_tensor_by_name("inputs/x-input:0")
	y = sess.graph.get_tensor_by_name("inputs/y-input:0")
	accuracy = sess.graph.get_tensor_by_name("score/accuracy:0")
	inference = sess.graph.get_tensor_by_name("inference/inference:0")

	feed = {x: X_test, y: y_test}
	print("Test set accuracy:", sess.run(accuracy, feed_dict=feed))

	# Additional report using scikit-learn
	predictions = sess.run(inference, feed_dict={x: X_test})
	print("Classification report:")
	print(metrics.classification_report(y_test.ravel(), predictions))
	print("Confusion matrix:")
	print(metrics.confusion_matrix(y_test.ravel(), predictions))