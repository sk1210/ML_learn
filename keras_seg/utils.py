   #utils
    import tensorflow as tf
import numpy as np
import argparse
import Models
import tensorflow as tf
from keras import backend as K


data_path = r"C:\Users\Z654281\Desktop\DATA\dataset\mapillary-vistas-dataset_zf_5k_subset/"

class_weights = np.array([0.2, 1, 1, 1,1])
class_weights = np.array([0.2, 1])

modelFns = {'aen':Models.aen.VGGSegnet ,
            'vgg_segnet':Models.VGGSegnet.VGGSegnet ,
            'vgg_segnet1':Models.VGGSegnet1.VGGSegnet ,
            'vgg_unet1':Models.VGGSegnet2.VGGSegnet,
            'vgg_unet2':Models.VGGUnet.VGGUnet2 ,
            'psp2':Models.VGGSegnet2.VGGSegnetPSP2,
            'psp':Models.VGGSegnet2.VGGSegnetPSP
            }

# VARIABLE MANIPULATION
def epsilon():

    _EPSILON = 1e-7
    return _EPSILON

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def categorical_crossentropy1(target, output, from_logits=False, axis=-1):

    print ("computing loss")
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))

    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output)*class_weights, axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

def categorical_crossentropy2(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

def readArgFromFile():
	args_list = []

	with open("train_args.cfg") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if "#" in line:
				continue
			if "=" in line:
				key, value = line.replace(" ","").split("=")

				key = "--" + key
				args_list.append(key)
				args_list.append(value)
			print (line,len(line))

	return args_list

def createParser():

	parser = argparse.ArgumentParser()
	parser.add_argument("--save_weights_path", type = str  )
	parser.add_argument("--train_images", type = str  )
	parser.add_argument("--train_annotations", type = str  )
	parser.add_argument("--n_classes", type=int )
	parser.add_argument("--input_height", type=int , default = 224  )
	parser.add_argument("--input_width", type=int , default = 224 )

	parser.add_argument('--validate',action='store_false')
	parser.add_argument("--val_images", type = str , default = "")
	parser.add_argument("--val_annotations", type = str , default = "")

	parser.add_argument("--epochs", type = int, default = 5 )
	parser.add_argument("--batch_size", type = int, default = 2 )
	parser.add_argument("--val_batch_size", type = int, default = 2 )
	parser.add_argument("--load_weights", type = str , default = "")

	parser.add_argument("--model_name", type = str , default = "")
	parser.add_argument("--optimizer_name", type = str , default = "adadelta")


	argsList = readArgFromFile()
	args = parser.parse_args(argsList)

	return args


def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def mean_iou(y_true, y_pred, smooth=1,axis=-2):
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    union = K.sum(y_true,axis) + K.sum(y_pred,axis) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou
