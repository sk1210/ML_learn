import Models , LoadBatches

label_dir = r"gdrive/My Drive/My_Projects/data/human_seg/ann/"
img_dir = r"gdrive/My Drive/My_Projects/data/human_seg/img/"
mask = r"gdrive/My Drive/My_Projects/data/human_seg/mask/"

# for multiple data
img_dir = r"gdrive/My Drive/My_Projects/data/All_converted/"

train_images_path = img_dir
train_segs_path = mask
train_batch_size = 4
n_classes = 2
input_height = 448
input_width = 448

save_weights_path = "gdrive/My Drive/My_Projects/weights/vgg_unet_448_"
epochs = 50
load_weights = ""
optimizer_name = "adadelta"
model_name = "vgg_unet"


modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet ,
             'vgg_unet':Models.VGGUnet.VGGUnet ,
             'vgg_unet2':Models.VGGUnet.VGGUnet2 ,
             'fcn8':Models.FCN8.FCN8 ,
             'fcn32':Models.FCN32.FCN32   }
