
import glob
import os
import itertools
import random
import cv2
from imgaug import augmenters as iaa
import numpy as np
import json
from matplotlib import pyplot as plt


class Labels:
    def __init__(self,configFile):
        self.configFile = configFile

        with open(configFile) as config_file:
            self.config = json.load(config_file)

        self.totalClasses = len(self.config)

        self.labelsMap = {
            5:1,
            15:2,
            36:3,
            37:4,
            55:5,
            #75:6,
        }
        self.labelsMap = {
            38: 1,
            36: 1,
            37: 1,
            # 55: 5,
            # 75:6,
        }
        self.labelsMap = {
            5:1,
            38:2,
            36:3,
            37:4,
            # 55: 5,
            # 75:6,
        }

        self.inverseMap = dict([(self.labelsMap[i], i) for i in self.labelsMap])
        self.default = 0
        self.default_color = (0,0,0)

    def LUT(self,value):

        if value in self.labelsMap:
            return self.labelsMap[value]
        return self.default

    def getColor(self,value):

        # inverse map
        if value == self.default:
            return self.default_color
        elif value == 3:
            return (0,0,255)
        assert value in self.inverseMap,"Unknown pixel"
        n = self.inverseMap[value]
        return self.config["labels"][n]["color"]


class ImageGeneratot:

    def __init__(self,images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,output_width):

        # set paths variable
        self.images_path = images_path
        self.segs_path = segs_path
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        # init iterator variables
        self.index = 0

        #load Images path
        self.train_index = -2
        self.test_index = 1000

        self.loadImages()


        # init all augmentor

        self.augmentor()

    def augmentor(self):
        self.commonAug = iaa.Sequential([
                    iaa.Fliplr(p=0.2)
            ])
        seq_det = self.commonAug.to_deterministic()

        # ------------only on image ------------------------

        sometimes = lambda aug: iaa.Sometimes(1, aug)

        self.seq_img_only = iaa.Sequential(
            [
                iaa.SomeOf((0, 1),
                           [

                               iaa.OneOf([
                                   #iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
                                   #iaa.AverageBlur(k=(2, 3)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   #iaa.MedianBlur(k=(3, 3)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               #iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1)),  # sharpen images


                               #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   #iaa.Dropout((0.001, 0.02), per_channel=0.3),  # randomly remove up to 10% of the pixels
                                   #iaa.CoarseDropout((0.03, 0.06), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),

                               # change brightness of images (by -10 to 10 of original value)
                               #iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                               #iaa.Multiply((0.8, 1.2), per_channel=0.5),

                               #iaa.ContrastNormalization((0.8, 1.5), per_channel=0.5),  # improve or worsen the contrast

                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        
    def readImages(self, folder_dir):
        folders = glob.glob(folder_dir + "/*")	
        img_dir = []
        label_dir = []
        for folder in folders:
            label_dir += glob.glob( os.path.join(folder, "mask") + "/*.png")
            img_dir += glob.glob( os.path.join(folder, "img") + "/*.png")
        return img_dir,label_dir
    
    def loadImages(self):
        self.imagesList,self.LabelsList =  self.readImages(self.images_path)
        #self.imagesList = glob.glob(self.images_path + "*.jpg") + glob.glob(self.images_path + "*.png") + glob.glob(self.images_path + "*.jpeg")
        #self.LabelsList = glob.glob(self.segs_path + "*.jpg") + glob.glob(self.segs_path + "*.png") + glob.glob(self.segs_path + "*.jpeg")
        
        self.imagesList.sort()
        self.LabelsList.sort()
        
        # filter blank images
        imgList , labelList = [],[]
        for img_name,label_name in zip(self.imagesList,self.LabelsList):
            label_img = cv2.imread(label_name,0)
            if cv2.countNonZero(label_img):
                imgList.append(img_name)
                labelList.append(label_name)
        print ("Blank Images : " , len(labelList), len(self.LabelsList))
        self.imagesList = imgList
        self.LabelsList = labelList
            
        assert len(self.imagesList) == len(self.LabelsList)

        for im, seg in zip(self.imagesList, self.LabelsList):
            assert (os.path.basename(im).split(".")[0] == os.path.basename(seg).split(".")[0])

        self.zippedList = list(zip(self.imagesList, self.LabelsList))

        self.zippedList_train = list(zip(self.imagesList[:self.train_index], self.LabelsList[:self.train_index]))

        # shuffle data
        random.seed(200)
        random.shuffle(self.zippedList_train)

    def returnPair(self):

        if self.index >= len(self.zippedList_train)-1: # reset index
            self.index = 0
            random.shuffle( self.zippedList_train )
        else:
            self.index += 1
            
        return self.zippedList_train[self.index]

    def getBatch(self):
        while True:

            X = []
            Y = []
            for _ in range(self.batch_size):
                img_name, seg_name = self.returnPair()
                X.append(img_name)
                Y.append(seg_name)

            # read batch and perform augmentation
            yield self.readBatch(X,Y)

    def readBatch(self,X,Y):

        images = [ cv2.imread(img) for img in X]
        seg_labels = [cv2.imread(seg,0) for seg in Y]

        # augment
        images, seg_labels = self.augment(images, seg_labels)

        images = np.array(images)
        seg_labels = np.array(seg_labels)
        # rescale and switch axis

        images = np.rollaxis(images, 3, 1)
        images = images / 255.0

        return images,seg_labels

    def augment(self,images,seg_labels):

        # resize
        images = [cv2.resize(img,(self.input_width,self.input_height),0,0,cv2.INTER_NEAREST) for img in images]
        seg_labels = [cv2.resize(img, (self.input_width, self.input_height), 0, 0, cv2.INTER_NEAREST) for img in seg_labels]

        images_aug = self.seq_img_only.augment_images(images)

        seq_det = self.commonAug.to_deterministic()
        images_aug_flipped = self.commonAug.augment_images(images_aug)
        seg_labels_augment = self.commonAug.augment_images(seg_labels)

        seg_labels_final = []

        for img in seg_labels_augment:
            seg_out = np.zeros((self.input_height, self.input_width, self.n_classes))
            seg_out[:, :, 0] += (img == 0).astype(int)
            seg_out[:, :, 1] += (img == 255).astype(int)
            
#             totalClass = 152
#             for i in range(totalClass):
#                 c = labels.LUT(i)
#                 seg_out[:, :, c] += (img == i).astype(int)

            seg_labels = np.reshape(seg_out, (self.output_width * self.output_height, self.n_classes))
            seg_labels_final.append(seg_labels)

        return images_aug_flipped, seg_labels_final

        # row, col = 2, self.batch_size * 2
        # fig = plt.figure(figsize=(row, col))
        # c = 0
        # for i, img in enumerate(images+seg_labels):
        #     c += 1
        #     fig.add_subplot(row, col, c)
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #
        # for i, img in enumerate(images_aug_flipped+seg_labels_augment):
        #     c += 1
        #     fig.add_subplot(row, col, c)
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()

        #print("debug")


# configFile = 'config.json'
# labels = Labels(os.path.join(data_path,configFile))

if __name__ == "__main__":

    args = createParser()
    # args_list = readArgFromFile()
    data_path = r"C:\Users\Z654281\Desktop\DATA\dataset\mapillary-vistas-dataset_zf_5k_subset/"

    train_images_path = data_path + args.train_images + "//"
    train_segs_path = data_path + args.train_annotations + "//"
    train_batch_size = args.batch_size
    n_classes = args.n_classes
    input_height = args.input_height
    input_width = args.input_width
    validate = False  # args.validate
    save_weights_path = args.save_weights_path
    epochs = args.epochs
    load_weights = args.load_weights
    output_width, output_height = input_width, input_height

    gen = ImageGeneratot(train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width)


    p = gen.getBatch()

    for i in range(10):
        next(p)

    print ("debug")
    
    
    
 
