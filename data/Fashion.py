import os
import pickle
import tensorflow as tf

imgPath = "/home/shahrukh/sk/dnn/tf/muti_label/ML_learn/data/images/"
pickle_file_path = "/home/shahrukh/sk/dnn/tf/muti_label/ML_learn/data/labels.pkl"
total_class = 228
w,h,c = 224,224,3
batch_size = 96

class fashion:
    def __init__(self):
        self.labels = []
        self.images = []

        # --------
        print ("loading data from file")
        self._loadLabels()
        self.iterator = self.createDataPipeLine()

        print("ready")

    def get_next(self):
        return self.iterator.get_next()

    def createDataPipeLine(self):
        ds = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        ds = ds.map(self.loadImage)
        ds = ds.batch(batch_size)

        iterator = ds.make_initializable_iterator()

        return iterator

    def _loadLabels(self,num_labels=10000):
        f = open(pickle_file_path, "rb")
        labels = pickle.load(f)
        f.close()

        fileNames = os.listdir(imgPath)
        self.labels = []
        self.images = []
        for i, img in enumerate(fileNames):
            id = int(img.split(".")[0])
            if not id in labels: continue

            one_hot = tf.one_hot(indices=labels[id], depth=total_class)
            self.labels.append(tf.reduce_sum(one_hot, reduction_indices=0))
            self.images.append(img)

            if i > num_labels: break
        return 1

    def loadImage(self,image,label):
        image_string = tf.read_file(imgPath + image)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [w, h])

        return (image_resized, label)

############################
# imgPath = "./../data/images/"
# fileNames = os.listdir("./../data/images/")
# fileNames = [ img for img in fileNames]
# total_classes = 228
# #one hot encod all data
#
# f = open("./../data/labels.pkl","rb")
# labels = pickle.load(f)
# f.close()
# multilabels = []
# images = []
# for i,img in enumerate(fileNames):
#     if i>10000:break
#     id  = int(img.split(".")[0])
#     if not id in labels:continue
#
#     one_hot = tf.one_hot(indices=labels[id], depth=total_classes)
#     multilabels.append(tf.reduce_sum(one_hot, reduction_indices=0))
#     images.append(img)
# ######################
#
# sess = tf.Session()
# #sess = tf.InteractiveSession()
#
# def my_func(input):
#     print (input)
#     one_hot = tf.one_hot(1, depth=total_classes)
#     return [2,2]
# def getLabels(image,label):
#     global labels
#
#     #tf.Print ("HELLO",data=[image])
#
#     image_string = tf.read_file(imgPath+image)
#     image_decoded =  tf.image.decode_jpeg(image_string)
#     image_resized = tf.image.resize_images(image_decoded, [28, 28])
#     #img = tf.decode_j
#     #label  = tf.py_func(my_func, [image], tf.string ,Tout = tf.int8)
#
#     #print ("ID:",sess.run(image))
#
#
#     #enc = labels[image]
#     #one_hot = tf.one_hot(indices=labels, depth=total_classes)
#     #train_data[d] = tf.reduce_sum(one_hot, reduction_indices=0)
#     #return (1, 1)
#     return(image_resized,label,image)
#
# temp= [0]*len(fileNames)
#
# ds = tf.data.Dataset.from_tensor_slices((images,multilabels))
# #ds.map(getLabels)
# ds = ds.map(getLabels)
# ds = ds.batch(32)
#
# iterator = ds.make_initializable_iterator()
#
#
# init = tf.global_variables_initializer()
# sess.run(init)
# # Initialize `iterator` with training data.
# #training_filenames = images
# sess.run(iterator.initializer)
# sess.run(iterator.get_next())


##################################