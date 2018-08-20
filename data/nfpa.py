import os
import pickle
import tensorflow as tf

imgPath = r"D:\shahrukh\MyScripts\DeepLearning\ML_learn\models\NFPA\images\\"
labelsPath  = r"D:\shahrukh\MyScripts\DeepLearning\ML_learn\models\NFPA\text\\"
#pickle_file_path = "/home/shahrukh/sk/dnn/tf/muti_label/ML_learn/data/labels.pkl"

total_class = 1+4
w,h,c = 224,224,3
batch_size = 64

class NFPA:
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
        ds = ds.repeat()

        iterator = ds.make_initializable_iterator()

        return iterator

    def readPos(self,file):
        f = open(file)
        pos = f.readline().strip().split(" ")
        return list(map(float, pos[1:]))

    def _loadLabels(self,num_labels=10000):
        fileNames = os.listdir(imgPath)
        fileNames = [name for name in fileNames if name.endswith(".jpg")]

        self.labels = []
        self.images = []
        for i, img in enumerate(fileNames):
            print (img)
            label = img.replace(".jpg",".txt")
            label  = self.readPos(imgPath + label)
            label.insert(0,1)
            self.labels.append(label)
            self.images.append(img)
            #print (img,label)

            if i > num_labels: break
        return 1

    def loadImage(self,image,label):
        image_string = tf.read_file(imgPath + image)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [w, h])

        #pos = self.convert_rev((h, w), label)
        return (image_resized, label)

    def convert_rev(self,size, box):
        cx,cy,cw,ch = box
        h,w = size
        left =  (cx - cw / 2.) * w
        right = (cx + cw / 2.) * w
        top =   (cy - ch / 2.) * h
        bot =   (cy + ch / 2.) * h
        box = map(int,[left,top,right,bot])
        return list(box)

if __name__ =="__main__":
    data = NFPA()