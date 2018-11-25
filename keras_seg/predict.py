import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
from arguments import *

images_path = img_dir
images_path = "gdrive/My Drive/My_Projects/data/human_seg/test_images/"
weights_path = "gdrive/My Drive/My_Projects/weights/vgg_segnet.14"
output_path = "gdrive/My Drive/My_Projects/results/"
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width )
m.load_weights(  weights_path )

m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

for i,imgName in enumerate(images):
	outName = imgName.replace( images_path ,  output_path )
	print (outName)
	X = LoadBatches.getImageArr(imgName , input_width  , input_height  )
	pr = m.predict( np.array([X]) )[0]
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
	seg_img = np.zeros( ( output_height , output_width , 3  ) )
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))
	img = cv2.imread(imgName)
	img = cv2.resize(img  , (input_width , input_height ))
	out_img = np.hstack((img, seg_img))
	cv2.imwrite(  outName , out_img )
	print (i)
	if i>100:break

