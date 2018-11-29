import argparse
import Models , LoadBatches
import LoadBatches1
import time
from arguments import *

def train():
	modelFN = modelFns[ model_name ]

	m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
	m.compile(loss='categorical_crossentropy',
		  optimizer= optimizer_name ,
		  metrics=['accuracy'])
	m.summary()

	load_weights =  "gdrive/My Drive/My_Projects/weights/vgg_unet_448_.110"
	epoch = 110
	if len( load_weights ) > 0:
		m.load_weights(load_weights, by_name=False,skip_mismatch=False)

	print ("Model output shape" ,  m.output_shape)

	output_height = m.outputHeight
	output_width = m.outputWidth

	#G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
	gen  = LoadBatches1.ImageGeneratot( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )	
	G = gen.getBatch()
	for ep in range( epoch+1,epochs ):
		print(ep , gen.index)
		
		m.fit_generator( G , 512  , epochs=1 )
		if ep % 2 == 0:
			m.save_weights( save_weights_path + "." + str( ep ) )
		time.sleep(3)
		#m.save( save_weights_path + ".model." + str( ep ) )

if __name__ == "__main__":
	train()
