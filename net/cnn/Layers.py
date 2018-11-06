# Types of Layers
#     Connected
#     Convolution
#     max-pool
#     batch_norm

import numpy as np
from py_cnn.utils import *


class Conv2d:

    def __init__(self, inputs, filters, kernel_size, stride=(1, 1), padding='same', activation=None):

        # init parameters
        self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.n_x, self.d_x, self.w_x, self.h_x = inputs.shape

        # set output parameter
        if padding == "same":
            self.output_width  = self.w_x
            self.output_height = self.h_x

        self.shape = (self.n_x, self.filters,self.output_width,self.output_height)
        # init weights
        self.W = np.random.randn(filters, self.d_x, self.kernel_size[0], self.kernel_size[1])

        #
        self.output = None
        self.h_output = None


    def forward(self, X):

        self.output = convolve3D(X, self.W)

class MaxPool:
    def __init__(self, inputs, kernel_size, stride=(1, 1), padding='valid'):
        self.inputs = inputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.n_x, self.d_x, self.w_x, self.h_x = inputs.shape

        self.shape = None
        self.output = None

    def forward(self):

        out_width, out_height = self.shape
        conv_out_mat = np.zeros((self.n_x, self.d_x , out_width, out_height))

        # iterate over all inputs_images ,N = batch _size
        for n in range(self.n_x):
                for d in range(self.d_x):
                    pass


#test
if __name__ == "__main__":

    testImg = cv2.imread(r"C:\Users\Z654281\PycharmProjects\dnn_learn\images\Page-2-Image-2.png")
    testImg = cv2.resize(testImg, (124, 124))

    # test Convolve3D
    x = np.array([testImg,testImg])

    x = x.swapaxes(1, 3)
    layer1 = Conv2d( x, 28, (5,5), activation="relu")
    layer2 = Conv2d( layer1, 16, (7, 7), activation="relu")

    layer1.forward(x)
    layer2.forward(layer1.output)

    output = layer1.output
    convolve3D(x, filter)



import tensorflow.layers as layer

layer.max_pooling2d()

