import os
import pickle
import tensorflow as tf

fileNames = "./../data/labels.pkl"
f= open(fileNames,"rb")
data = pickle.load(f)
f.close()


total_classes = 228
#one hot encod all data



def multiple_one_hot(cat_tensor, depth_list):

    one_hot_enc_tensor = tf.one_hot(cat_int_tensor[:,0], depth_list[0], axis=1)
    for col in range(1, len(depth_list)):
        add = tf.one_hot(cat_int_tensor[:,col], depth_list[col], axis=1)
        one_hot_enc_tensor = tf.concat([one_hot_enc_tensor, add], axis=1)

    return one_hot_enc_tensor

def splitImgLabels(imageName):
    print (imageName)
    imgId = str(imageName)+".jpg"
    #labels = data[imageName]
    return (imgId),[0]*10



######################

sess = tf.Session()

train_data = {}
for d in data:
    one_hot = tf.one_hot(indices = data[d],depth= total_classes)
    train_data[d] = tf.reduce_sum(one_hot, reduction_indices=0)

ds = tf.data.Dataset.from_tensor_slices(train_data)
ds.map(splitImgLabels)

print ((data.keys))
