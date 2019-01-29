from builtins import range
import numpy as np
# from skimage.io import imread
import tensorflow as tf
import sys
import cv2


def main():
    # Read in image with the shape (rows, cols, channels)
    im = cv2.imread('./img/face.png')

    im = np.array(im) / 255.

    invSpatialStdev = float(1. / 5.)
    invColorStdev = float(1. / .125)

    rows = im.shape[0]
    cols = im.shape[1]
    nb_points = rows * cols
    # Construct the position vectors out of x, y, r, g, and b.
    positions = np.zeros((nb_points, 5), dtype='float32')
    color = np.zeros((nb_points, 3), dtype='float32')
    output = np.zeros(im.shape, dtype="float32")

    for r in range(rows):
        for c in range(cols):
            positions[cols * r + c, 0] = invSpatialStdev * c
            positions[cols * r + c, 1] = invSpatialStdev * r
            positions[cols * r + c, 2] = invColorStdev * im[r, c, 0]
            positions[cols * r + c, 3] = invColorStdev * im[r, c, 1]
            positions[cols * r + c, 4] = invColorStdev * im[r, c, 2]

            color[cols * r + c, 0] = im[r, c, 0]
            color[cols * r + c, 1] = im[r, c, 1]
            color[cols * r + c, 2] = im[r, c, 2]

    lattice_fllter_module = tf.load_op_library("./my_op.so")

    position_tensor = tf.placeholder(dtype=tf.float32, shape=positions.shape, name="pos_input")
    value_tensor = tf.placeholder(dtype=tf.float32, shape=color.shape, name="val_input")

    out_tensor = lattice_fllter_module.lattice_filter(value_tensor, position_tensor)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        out_ = sess.run(out_tensor, feed_dict={position_tensor: positions, value_tensor: color})

        for r in range(rows):
            for c in range(cols):
                # print("--------")
                output[r, c, 0] = out_[cols * r + c, 0]
                output[r, c, 1] = out_[cols * r + c, 1]
                output[r, c, 2] = out_[cols * r + c, 2]

        output -= output.min()
        output /= output.max()
        cv2.imwrite("./results/filtered_face.png", (255 * output).astype(np.uint8))


if __name__ == '__main__':
    main()
