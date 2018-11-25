import Models , LoadBatches

label_dir = r"/home/shahrukh/Files/Data/person_seg__ds9/ann/"
img_dir = r"/home/shahrukh/Files/Data/person_seg__ds9/img/"
mask = r"/home/shahrukh/Files/Data/person_seg__ds9/mask/"

train_images_path = img_dir
train_segs_path = mask
train_batch_size = 1
n_classes = 2
input_height = 640
input_width = 480

save_weights_path = "weights/"
epochs = 50
load_weights = ""
optimizer_name = "adadelta"
model_name = "vgg_segnet"


modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet ,
             'vgg_unet':Models.VGGUnet.VGGUnet ,
             'vgg_unet2':Models.VGGUnet.VGGUnet2 ,
             'fcn8':Models.FCN8.FCN8 ,
             'fcn32':Models.FCN32.FCN32   }
