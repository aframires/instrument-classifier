from __future__ import absolute_import, print_function, division

import tensorflow as tf


class ModelsParentClass():

    def read_from_tfrecord(self, data_record):
        tfrecord_features = tf.parse_single_example(data_record, features={'spec': tf.FixedLenFeature([], tf.string),
                                                                           'shape': tf.FixedLenFeature([], tf.string),
                                                                           'label': tf.FixedLenFeature([], tf.int64)},
                                                    name='features')
        shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
        spec = tf.decode_raw(tfrecord_features['spec'], tf.float32)
        spec = tf.reshape(spec, shape)
        label = tfrecord_features['label']
        return spec, label

    def compute_accuracy(self, logits, labels, nb_classes):
        probs = tf.nn.softmax(logits)
        prediction = tf.argmax(probs, 1)
        equality = tf.equal(tf.cast(prediction, tf.int64), labels)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        cm = tf.confusion_matrix(labels, prediction, num_classes=nb_classes)
        return accuracy, cm

    def print_message(self):
        print("Works so far")


