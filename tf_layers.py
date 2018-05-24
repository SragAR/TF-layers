import tensorflow as tf
import cv2

imageObj = cv2.imread('input.jpg')
imageObj = imageObj[:,:,::-1]
image = tf.convert_to_tensor(imageObj,dtype=tf.float32)
input_layer = tf.reshape(image, [-1,720, 1280, 3])
pool1 = tf.layers.average_pooling2d(inputs=input_layer, pool_size=[2, 2], strides=2)
pool2 = tf.layers.average_pooling2d(inputs=pool1, pool_size=[2, 2], strides=2)
pool3 = tf.layers.average_pooling2d(inputs=pool2, pool_size=[2, 2], strides=2)
pool4 = tf.layers.average_pooling2d(inputs=pool3, pool_size=[2, 2], strides=2)

resized = tf.image.resize_images(pool4, [720, 1280])
output_image = tf.image.encode_jpeg(tf.cast(resized[0], tf.uint8))

file_name = tf.constant('./Ouput_image.jpeg')
with tf.Session():
      tf.write_file(file_name, output_image).run()
