
import numpy as np
import tensorflow as tf
from bottle.Bottle import BottleData
from fashion.Fashion import fashion
import os
import math
from DogBreed import DogBreedData
from fashion import Fashion
import DogBreed


train = 0

w,h,c = 224,224,3
batch_size = Fashion.batch_size
total_class = Fashion.total_class

# w,h,c = 224,224,3
# batch_size = DogBreed.batch_size
# total_class = DogBreed.total_class

X = tf.placeholder(tf.float32, [None, w, h, c])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, total_class])

#def myModel(features,labels,mode):
"""Model function for CNN."""
# Input Layer
input_layer = tf.reshape(X, [-1, w, h, c])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=96,
    kernel_size=[7,7],
    padding="same",
    activation=tf.nn.relu,strides=4)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Convolutional Layer #3
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=384,
    kernel_size=[3 ,3],
    padding="same",
    activation=tf.nn.relu, strides=1)

conv4 = tf.layers.conv2d(
    inputs=conv3,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu, strides=1)



conv5 = tf.layers.conv2d(
    inputs=conv4,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu, strides=1)

pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

# Dense Layer
pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 256])


dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
dropout1 = tf.layers.dropout(
    inputs=dense1, rate=0.5)#, training=mode == tf.estimator.ModeKeys.TRAIN)

dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)
dropout2 = tf.layers.dropout(
    inputs=dense2, rate=0.5)

# Logits Layer
Ylogits = tf.layers.dense(inputs=dropout2, units=total_class)

#Y = tf.nn.softmax(Ylogits)
Y = tf.nn.sigmoid(Ylogits)

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y_)
#cross_entropy = tf.reduce_mean(cross_entropy)*100

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = Ylogits, labels = Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
#correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(Ylogits)), tf.round(Y_))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

step = tf.placeholder(tf.int32)
lr = 0.01 +  tf.train.exponential_decay(0.0005, step, 2000, 1/math.e)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#----------------------------#
# start session and init variables
sess = tf.Session()
#data = DogBreedData()
data = fashion()

next_batch =  data.iterator.get_next()
init_var = tf.variables_initializer([data.iterator])
init = tf.global_variables_initializer()

sess.run(init_var)
sess.run(init)
#print(sess.run(pool5.shape))
update_train_data  = 1

batch_X_test, batch_Y_test = sess.run(data.get_next())

# tf save model
saver = tf.train.Saver()

saver.restore(sess, "./my_model_sig")
print("Model restored.")

if train:
    # train loop
    for i in range(10000):
    #def train(i):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = sess.run(data.get_next())

        # compute training values for visualisation

        if update_train_data:
            a, c, l = sess.run([accuracy, cross_entropy, lr],
                                      feed_dict={X: batch_X, Y_: batch_Y, step: i})
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")


        # Compute Test Values For Visualisation
        if i%10 == 0 :
            a, c = sess.run([accuracy, cross_entropy],
                                feed_dict={X: batch_X_test, Y_: batch_Y_test})
            print(str(i) + ": ********* epoch " + str(i*100//len(data.labels)) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

        if i % 100 == 0:
            saver.save(sess, './my_model_sig')

        #sess.run(tf.argmax(Y))
        # the backpropagation training step
        sess.run(train_step, {X: batch_X, Y_: batch_Y})

        #print(sess.run([Y,Y_],feed_dict={X:batch_X,Y_: batch_Y_test}))

    #datavis.animate(train_step, 10001, train_data_update_freq=10, test_data_update_freq=100)
else:
    import cv2

    img = cv2.imread("samples/1.jpg")
    img = cv2.resize(img,(w,h))

    input = [img]

    Y_out = sess.run([Y],feed_dict={X:input})
    print (Y_out)
    print (max(Y_out[0][0]*100),np.argmax(Y_out))

    # write results
    result = []

    thresh = 0.6
    for i,x in enumerate(Y_out[0][0],1):
        if x > thresh:
            result.append(i)

    print (result)
    
    #generate csv file
    testImgPath = "/home/shahrukh/sk/dnn/tf/muti_label/ML_learn/data/images/"

    images = os.listdir(testImgPath)
    images = sorted(images, key=lambda x: int(x.replace(".jpg","")) )

    f = open("result.csv","w")

    for i,imgName in enumerate(images):
        if not imgName.endswith(".jpg"):continue

        img = cv2.imread(testImgPath+"//"+imgName)
        img = cv2.resize(img, (w, h))

        input = [img]

        Y_out = sess.run([Y], feed_dict={X: input})
        result = []
        for j, x in enumerate(Y[0][0], 0):
            if x > thresh:
                result.append(str(j))
        f.write(imgName.replace(".jpg",",")+" ".join(result)+"\n")

        if i%100 : print (i)
        #if i>100:break

    f.close()
        # write line