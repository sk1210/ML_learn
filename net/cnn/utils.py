import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import measure

def convolve3D( x, filter, pad = 1, stride = 1 ):

    # padded_input = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')

    N, C, W, H = x.shape

    nf, c, f_w, f_h = filter.shape

    # check shape
    assert (c == C)
    assert (H + 2 * pad - f_h) % stride == 0
    assert (W + 2 * pad - f_w) % stride == 0

    out_height = int((H + 2 * pad - f_h) / stride + 1)
    out_width  = int((W + 2 * pad - f_w) / stride + 1)

    out_width, out_height = W, H

    conv_out_mat = np.zeros((N, nf, C, out_width, out_height))

    # iterate over all inputs_images ,N = batch _size
    for n in range(N):
        for f in range(nf):
            for c in range(C):
                conv_out_mat[n][f][c] = convolve2d(x[n][c][:, :], filter[f][c], mode='same')

    conv_output = np.sum(conv_out_mat, axis=2)/(f_h * f_w * c)

    print("Convolve 3D Debug")

    return conv_output

def maxPool(x,block_size):

    row,col = x.shape
    block_size = 2

    index_image = np.zeros(x.shape)
    max_pool_out = np.zeros((row//block_size,col//block_size))

    index_map2 = {0 : (0,0),1:(0,1),2:(1,0),3:(1,1)}
    for i in range(0,row-block_size,block_size):
        for j in range(0,col-block_size,block_size):
            block = x[i:i+block_size, j:j+block_size]
            max_index = block.flatten().argmax()
            max_value = np.max(block)

            px,py = index_map2[max_index]
            index_image[i:i+block_size, j:j+block_size].itemset(px,py,1)

            max_pool_out.itemset(i//2,j//2,max_value)

    max_value_image = x.copy()
    max_value_image[np.where(index_image==0)] = 0

    #measure.block_reduce(x,block_size,func=np.max)

    return index_image,max_pool_out

if __name__ == "__main__":

    # ______________________ Test Convolution ________________________

    testImg = cv2.imread(r"C:\Users\Z654281\PycharmProjects\dnn_learn\images\Page-2-Image-2.png")
    testImg = cv2.resize(testImg,(31,31))

    # # test Convolve3D
    # x = np.array([testImg])
    # filter = np.random.random((1,3,5,5))
    # convolve3D(x,filter)

    # ______________________ Test Pooling ________________________

    index_image, max_pool_out = maxPool( testImg[:,:,0].astype(np.float32)/255 , 2)
    print("Debug")
