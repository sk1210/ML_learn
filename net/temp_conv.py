import numpy as np
import cv2
from scipy.signal import convolve2d
testImg = cv2.imread(r"C:\Users\Z654281\PycharmProjects\dnn_learn\images\Page-2-Image-2.png")
testImg = cv2.resize(testImg,(124,124))
X = np.array(testImg).astype(np.float32)/255
Weight = np.zeros((1,3,3,3))


Weight[0][0] = np.array([[ 1.,  0., -1.],
                    [ 1.,  0., -1.],
                    [ 1.,  0., -1.]])

Weight[0][1] = np.array([[ 1.,  0., 1.],
                    [ 0.,  0., 0.],
                    [ -1.,  0., -1.]])

#perform reshape
#cv2.imshow("img",testImg)
#cv2.waitKey()
cv2.destroyAllWindows()

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  out_height = int(out_height)
  out_width  = int(out_width)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

  return cols

def convolve( input, f_h ,f_w , pad = 1, stride = 1 ):

    padded_input = np.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')

    N, H, W, C = input.shape

    # check shape
    assert (H + 2 * pad - f_h) % stride == 0
    assert (W + 2 * pad - f_w) % stride == 0

    out_height = int((H + 2 * pad - f_h) / stride + 1)
    out_width  = int((W + 2 * pad - f_w) / stride + 1)

    conv_out = np.zeros((out_width, out_height))

    conv_out_mat = np.zeros((C, out_width, out_height))
    for i in range(C):
        conv_out_mat[i] = convolve2d(input[0][:, :, i], Weight[0][i], mode='same')

    filter_out = np.sum(conv_out_mat, axis=0)

    print("Debug")


small = testImg#cv2.resize(testImg,(8,8))
input = np.array([small])

convolve(input , 3,3,1,1)

r,g,b = cv2.split(testImg)

print("debug")


