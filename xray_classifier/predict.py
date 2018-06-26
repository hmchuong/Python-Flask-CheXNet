import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import xray_classifier.scripts.inception_preprocessing as inception_preprocessing
from xray_classifier.scripts.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
import pdb
from PIL import ImageFile
from PIL import Image
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

slim = tf.contrib.slim
labels = {0: "choose_bse", 1: "choose_origin"}

width = 299
height = 299
dir_path = os.path.dirname(os.path.realpath(__file__))

#Get the latest checkpoint file
checkpoint_file = os.path.join(dir_path,'model/model.ckpt-6836')

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB png data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_image(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class ChoosingMethodModel():
    def __init__(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            self.image_reader = ImageReader()
            tf.logging.set_verbosity(tf.logging.INFO)

            self.image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3))
            self.image = inception_preprocessing.preprocess_image(self.image_placeholder, height, width, is_training = False)


            #Now create the inference model but set is_training=False
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, self.end_points = inception_resnet_v2(tf.expand_dims(self.image, 0), num_classes = 2, is_training = False)

            # #get all the variables to restore from the checkpoint file and create the saver function to restore
            exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            def restore_fn(sess):
                return saver.restore(sess, checkpoint_file)

            #Just define the metrics to track without the loss or whatsoever
            self.predictions = tf.argmax(self.end_points['Predictions'], 1)
            #accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            #metrics_op = tf.group(accuracy_update)

            #Create the global step and an increment op for monitoring
            global_step = get_or_create_global_step()
            global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
            self.sv = tf.train.Supervisor(logdir = None, summary_op = None, saver = None, init_fn = restore_fn)#tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_file)
            self.sess = self.sv.prepare_or_wait_for_session(wait_for_checkpoint=True, max_wait_secs=7200)


    def predict(self, image_path):
        # tf.reset_default_graph()
        # with tf.Graph().as_default() as graph:
        #     self.image_reader = ImageReader()
        #     tf.logging.set_verbosity(tf.logging.INFO)
        #
        #     self.image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3))
        #     image = inception_preprocessing.preprocess_image(self.image_placeholder, height, width, is_training = False)
        #
        #
        #     #Now create the inference model but set is_training=False
        #     with slim.arg_scope(inception_resnet_v2_arg_scope()):
        #         logits, self.end_points = inception_resnet_v2(tf.expand_dims(image, 0), num_classes = 2, is_training = False)
        #
        #     # #get all the variables to restore from the checkpoint file and create the saver function to restore
        #     exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        #     variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
        #     saver = tf.train.Saver(variables_to_restore)
        #     def restore_fn(sess):
        #         return saver.restore(sess, checkpoint_file)
        #
        #     #Just define the metrics to track without the loss or whatsoever
        #     self.predictions = tf.argmax(self.end_points['Predictions'], 1)
        #     #accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        #     #metrics_op = tf.group(accuracy_update)
        #
        #     #Create the global step and an increment op for monitoring
        #     global_step = get_or_create_global_step()
        #     global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        #     sv = tf.train.Supervisor(logdir = None, summary_op = None, saver = None, init_fn = restore_fn)#tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_file)
        result = 0
        #     with sv.managed_session() as sess:
        # with self.sv.managed_session() as sess:
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        image = self.image_reader.decode_png(self.sess, image_data)

        _image, _predict, prob, logits = self.sess.run([self.image, self.predictions, self.end_points['Predictions'], self.end_points['Logits']], feed_dict={self.image_placeholder: image})
        print(_predict, prob)
        result = float(prob[0][0])
        return result
