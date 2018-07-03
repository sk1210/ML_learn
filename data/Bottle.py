import os
import pickle
import tensorflow as tf
from os.path import join
imgPath = "bottle/images/"

total_class = 4

w,h,c = 224,224,3
batch_size = 128

class BottleData():
    def __init__(self):
        self.labels = []
        self.images = []

        # --------
        print (" Loading Data From File ")
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
        folders = os.listdir(imgPath)

        for i,f in enumerate(folders):

            if f not in ['0','1','2','3']:continue
            print(i, f)
            for name in os.listdir(join(imgPath,f)):
                if not name.endswith(".jpg"): continue
                self.images.append(join(imgPath,f,name))
                self.labels.append(int(f))
        #random.shuffle(a)
    def loadImage(self,image,label):
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [w, h])
        label = tf.one_hot(label, depth= total_class)
        return (image_resized, label)

if __name__ =="__main__":
    sess = tf.Session()
    data = BottleData()

    init_var = tf.initialize_variables([data.iterator])
    init = tf.global_variables_initializer()

    sess.run(init_var)

    # start training
    print (sess.run(data.get_next()))