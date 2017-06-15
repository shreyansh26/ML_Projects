from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 

# Set parameters
learning_rate = 0.01
training_iteration = 10
batch_size = 100
display_step = 2

# TF graph input
x = tf.placeholder("float", [None, 784])    # MNIST data image of shape 28*28 = 784
y = tf.placeholder("float", [None, 10])     # 0-9 digits recognition => 10 classes

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
	# Construct a linear model
	model = tf.nn.softmax(tf.matmul(x, W) + b)   # Softmax

# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function") as scope:
	# Minimise error using cross entropy
	# Cross entropy
	cost_function = -tf.reduce_sum(y*tf.log(model))

	# Create a scalar summary
	tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
	# Gradient Descent
	optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initialise the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merge_summary = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter('digit_log', graph_def=sess.graph_def)

    for iteration in range(training_iteration):

    	avg_cost = 0

    	total_batch = int(mnist.train.num_examples/batch_size)

    	# Loop over all batches
    	for i in range(total_batch):
    		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    		# Fit training using batch data
    		sess.run(optimiser, feed_dict={x:batch_xs, y:batch_ys})
    		# Compute average loss
    		avg_cost += sess.run(cost_function, feed_dict={x:batch_xs, y:batch_ys})/total_batch

    		# Write logs for each iteration
    		summary_str = sess.run(merge_summary, feed_dict={x:batch_xs, y:batch_ys})
    		summary_writer.add_summary(summary_str, iteration*total_batch + i)

    	if iteration%display_step == 0:
    		print("Iteration: %04d, Cost: %.9f" %(iteration+1, avg_cost))

    print("Training complete")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    #tf.Print(model)



