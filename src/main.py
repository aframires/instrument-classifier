
from __future__ import absolute_import,print_function,division

import tensorflow as tf

#from models.model_basic_cnn import ModelBasicCNN
#from models.singleLayerMM2 import SingleLayerMM2
from models.singleLayerMM3 import SingleLayerMM3

print('Whatsup man')
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 150, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 50, 'Batch size.  '
                         'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_data', "/homedtic/aramires/NSynth/FewFeatures/nsynth-train-spec.tfrecord", 'Directory to put the training data.')
flags.DEFINE_string('valid_data', "/homedtic/aramires/NSynth/FewFeatures/nsynth-valid-spec.tfrecord"
, 'Directory to put the valid data.')
flags.DEFINE_boolean('is_training', True, 'If true, selected model trains')
flags.DEFINE_string('model_name','single_layer_mm3','Name of the model for training or inference')
flags.DEFINE_integer('nb_classes',11,'Number of classes for the classifier')
flags.DEFINE_integer('valid_iters',252,'Number of valid iterations')
flags.DEFINE_integer('train_iters',4584,'Number of train iterations')
flags.DEFINE_string('dataset','nofx','Datasets to use for augmentation (all or specific effect')


print(FLAGS)

"""
if(FLAGS.model_name=='model_basic_cnn'):
    if(FLAGS.is_training):
        #model = ModelBasicCNN(FLAGS.learning_rate,FLAGS.is_training,FLAGS.nb_classes,FLAGS.num_epochs,FLAGS.train_iters,FLAGS.valid_iters,FLAGS.batch_size,FLAGS.train_data,FLAGS.valid_data,FLAGS.dataset)
        #model.test_load_from_tfrecords()
        #model.train()

if(FLAGS.model_name=='single_layer_mm2'):
    if(FLAGS.is_training):
        #model = SingleLayerMM2(FLAGS.learning_rate,FLAGS.is_training,FLAGS.nb_classes,FLAGS.num_epochs,FLAGS.train_iters,FLAGS.valid_iters,FLAGS.batch_size,FLAGS.train_data,FLAGS.valid_data,FLAGS.dataset)
        #model.test_load_from_tfrecords()
        #model.train()
"""
if(FLAGS.model_name=='single_layer_mm3'):
    if(FLAGS.is_training):
        print(FLAGS.valid_data)
        print(FLAGS.train_data)
        model = SingleLayerMM3(FLAGS.learning_rate,FLAGS.is_training,FLAGS.nb_classes,FLAGS.num_epochs,FLAGS.train_iters,FLAGS.valid_iters,FLAGS.batch_size,FLAGS.train_data,FLAGS.valid_data,FLAGS.dataset)
        #model.test_load_from_tfrecords()
        model.train()
