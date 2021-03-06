import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import sys
from arguments import *
from matplotlib import pyplot as plt
from IPython.display import Image
import time

#images_path = img_dir
images_path = "gdrive/My Drive/My_Projects/data/human_seg/test_images/"
images_path = "gdrive/My Drive/My_Projects/data/human_seg/img/"
weights_path = "gdrive/My Drive/My_Projects/weights/vgg_unet_448_."
output_path = "gdrive/My Drive/My_Projects/results/"
modelFN = modelFns[ model_name ]


weights_path += sys.argv[1]
m = modelFN( n_classes , input_height=input_height, input_width=input_width )
m.load_weights(  weights_path )

m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg" ) +glob.glob( images_path + "*.JPG" )
images.sort()
print(images)
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
	blend_img =  img.copy()
	blend_img[:,:,2] += pr*100
	out_img = np.hstack((img, blend_img))
	
	cv2.imwrite(  outName , out_img )
	print (i)
	plt.title('my picture'+str(i))
	plt.imshow(img)
	plt.show();
	cv2.imwrite("test.jpg",out_img)
	time.sleep(1)
	display("test.jpg")
	time.sleep(0.5)
	if i>100:break

