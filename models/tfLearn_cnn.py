from __future__ import print_function

import tensorflow as tf
import tflearn
import loadData
# --------------------------------------
# High-Level API: Using TFLearn wrappers
# --------------------------------------

# Using MNIST Dataset

w,h,c = 124,124,3
batch_size = loadData.batch_size
total_class = 228

# User defined placeholders
with tf.Graph().as_default():
    # Placeholders for data and labels
    X = tf.placeholder(shape=(None, w,h,c), dtype=tf.float32)
    Y = tf.placeholder(shape=(None, total_class), dtype=tf.float32)

    net = tf.reshape(X, [-1, w, h, c])

    # Using TFLearn wrappers for network building
    net = tflearn.conv_2d(net, 32, 7, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.dropout(net, 0.8)

    net = tflearn.conv_2d(net, 64, 5, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.dropout(net, 0.8)


    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 128, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 256, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, total_class, activation='sigmoid')

    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


    #init data
    data = loadData.fashion()
    init_var = tf.initialize_variables([data.iterator])
    # Initializing the variables
    init = tf.global_variables_initializer()


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_var)
        next_element = data.iterator.get_next()
        for epoch in range(1000):  # 2 epochs
            avg_cost = 0.
            total_batch = 1 #int(mnist_data.train.num_examples / batch_size)
            for i in range(total_batch):
                print (data.get_next())
                batch_xs, batch_ys = sess.run(next_element) #mnist_data.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
                cost = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys})
                #print (batch_ys)

                avg_cost += cost / total_batch
                if i % 5 == 0:
                    print("Epoch:", '%03d' % (epoch + 1), "Step:", '%03d' % i,"Avg Loss:", str(cost))


