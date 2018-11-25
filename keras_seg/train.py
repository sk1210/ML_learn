import argparse
import Models , LoadBatches

from arguments import *

def train():
	modelFN = modelFns[ model_name ]

	m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
	m.compile(loss='categorical_crossentropy',
		  optimizer= optimizer_name ,
		  metrics=['accuracy'])


	if len( load_weights ) > 0:
		m.load_weights(load_weights)

	print ("Model output shape" ,  m.output_shape)

	output_height = m.outputHeight
	output_width = m.outputWidth

	G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


	for ep in range( epochs ):
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )

if __name__ == "__main__":
	train()