import numpy as np
import cv2
from scipy.signal import convolve2d


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

    conv_output = np.sum(conv_out_mat, axis=2)

    print("Debug")

    return conv_output


if __name__ == "__main__":

    # ______________________ Test Convolution ________________________

    testImg = cv2.imread(r"C:\Users\Z654281\PycharmProjects\dnn_learn\images\Page-2-Image-2.png")
    testImg = cv2.resize(testImg,(124,124))

    # test Convolve3D
    x = np.array([testImg])
    filter = np.random.random((1,3,5,5))
    convolve3D(x,filter)

    print ("Debug")
