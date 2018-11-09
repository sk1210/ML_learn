# Types of Layers
#     Connected
#     Convolution
#     max-pool
#     batch_norm

import numpy as np
from py_cnn.utils import *
from skimage import measure
from py_cnn.activation import *

class Conv2d:

    def __init__(self, inputs, filters, kernel_size, stride=(1, 1), padding='same', activation=None):

        # init parameters
        self.type = "Convolution"
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

        if self.activation =="relu":
            self.h_output = relu(self.output)

        return self.h_output

class MaxPool:

    def __init__(self, inputs, kernel_size, stride=(1, 1), padding='valid'):

        self.type = "MaxPool"
        self.inputs = inputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.n_x, self.d_x, self.w_x, self.h_x = inputs.shape

        self.shape = self.n_x, self.d_x,self.w_x //kernel_size , self.h_x // kernel_size
        self.output = None

    def forward(self, X):

        n, d, width, height = X.shape
        out_width  = width // self.kernel_size
        out_height = height // self.kernel_size

        conv_out_mat = np.zeros((self.n_x, self.d_x , out_width, out_height))
        conv_index_mat = np.zeros((self.n_x, self.d_x , width, height))
        # iterate over all inputs_images ,N = batch _size
        for n in range(self.n_x):
            for d in range(self.d_x):
                conv_index_mat[n][d], conv_out_mat[n][d] = maxPool(X[n][d], self.kernel_size)
                #print ("Debug")

        self.output = conv_out_mat

#test
if __name__ == "__main__":

    testImg = cv2.imread(r"C:\Users\Z654281\PycharmProjects\dnn_learn\images\Page-2-Image-2.png")
    testImg = cv2.resize(testImg, (8, 8))

    # test Convolve3D
    x = np.array([testImg,testImg])

    x = x.swapaxes(1, 3)
    layer1 = Conv2d( x/255, 28, (5,5), activation="relu")
    layer2 = Conv2d( layer1, 16, (3, 3), activation="relu")
    layer3 = MaxPool(layer2, 2)

    layer4 = Conv2d(layer3, 28, (3, 3), activation="relu")
    layer5 = Conv2d(layer4, 16, (3, 3), activation="relu")
    layer6 = MaxPool(layer5, 2)

    layers = [layer1,layer2,layer3,layer4, layer5, layer6]
    layer1.forward(x/255)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    layer4.forward(layer3.output)
    layer5.forward(layer4.output)
    layer6.forward(layer5.output)

    for layer in layers:
        print (layer.inputs.shape[1:],layer.type + "  -> ", layer.shape[1:])
    output = layer1.output
    convolve3D(x, filter)
