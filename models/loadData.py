import os
import pickle
import tensorflow as tf

imgPath = "./../data/train/"
fileNames = os.listdir("./../data/train")
fileNames = [ img for img in fileNames]
total_classes = 228
#one hot encod all data

f = open("./../data/labels.pkl","rb")
labels = pickle.load(f)
f.close()
multilabels = []
images = []
for i,img in enumerate(fileNames):
    if i>10000:break
    id  = int(img.split(".")[0])
    if not id in labels:continue

    one_hot = tf.one_hot(indices=labels[id], depth=total_classes)
    multilabels.append(tf.reduce_sum(one_hot, reduction_indices=0))
    images.append(img)
######################

sess = tf.Session()
#sess = tf.InteractiveSession()

def my_func(input):
    print (input)
    one_hot = tf.one_hot(1, depth=total_classes)
    return [2,2]
def getLabels(image,label):
    global labels

    #tf.Print ("HELLO",data=[image])

    image_string = tf.read_file(imgPath+image)
    image_decoded =  tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    #img = tf.decode_j
    #label  = tf.py_func(my_func, [image], tf.string ,Tout = tf.int8)

    #print ("ID:",sess.run(image))


    #enc = labels[image]
    #one_hot = tf.one_hot(indices=labels, depth=total_classes)
    #train_data[d] = tf.reduce_sum(one_hot, reduction_indices=0)
    #return (1, 1)
    return(image_resized,label,image)

temp= [0]*len(fileNames)

ds = tf.data.Dataset.from_tensor_slices((images,multilabels))
#ds.map(getLabels)
ds = ds.map(getLabels)
ds = ds.batch(32)

iterator = ds.make_initializable_iterator()


init = tf.global_variables_initializer()
sess.run(init)
# Initialize `iterator` with training data.
#training_filenames = images
sess.run(iterator.initializer)
print(sess.run(iterator.get_next()))
