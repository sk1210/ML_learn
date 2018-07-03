import os
import csv
import pickle
import tensorflow as tf
from os.path import join
imgPath = "D:\shahrukh\MyScripts\DeepLearning\ML_learn\data\dog_breed/train/"
labels_csv = r"D:\shahrukh\MyScripts\DeepLearning\ML_learn\data\dog_breed\labels.csv"
label_names = "D:\shahrukh\MyScripts\DeepLearning\ML_learn\data\dog_breed\dog_names.txt"

total_class = 120
w,h,c = 224,224,3
batch_size = 8

class DogBreedData():
    def __init__(self):
        self.labels = []
        self.images = []

        # --------
        print (" Loading Data From File ")
        self.readNames()
        self._loadLabels()
        self.iterator = self.createDataPipeLine()

    def get_next(self):
        return self.iterator.get_next()

    def createDataPipeLine(self):
        ds = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        ds = ds.shuffle(buffer_size=len(self.labels))
        ds = ds.map(self.loadImage)
        ds = ds.repeat()
        ds = ds.batch(batch_size)

        iterator = ds.make_initializable_iterator()

        return iterator

    def _loadLabels(self,num_labels=10000):


        labels_dict = {}
        f = open(labels_csv)
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if i == 0: continue
            # print (row)
            labels_dict[row[0]] = row[1]

        images = os.listdir(imgPath)

        for name in images:
            if not name.endswith(".jpg"): continue
            self.images.append(join(imgPath,name))
            label = labels_dict[name.replace(".jpg","")]
            label_id = self.names_dict[label]
            self.labels.append(int(label_id))
        #random.shuffle(a)
    def loadImage(self,image,label):
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [w, h])
        label = tf.one_hot(label, depth= total_class)
        return (image_resized, label)

    def readNames(self):
        label_names = "D:\shahrukh\MyScripts\DeepLearning\ML_learn\data\dog_breed\dog_names.txt"
        f = open(label_names)
        names = f.readlines()
        print (names)
        names = [name.strip("\n") for name in names]
        print(names)
        f.close()
        self.names_dict = dict([ (x,i) for i,x in enumerate(names) ] )


if __name__ =="__main__":
    sess = tf.Session()
    data = DogBreedData()

    init_var = tf.initialize_variables([data.iterator])
    init = tf.global_variables_initializer()

    sess.run(init_var)

    # start training
    print (sess.run(data.get_next()))