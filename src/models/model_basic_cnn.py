from __future__ import absolute_import,print_function,division

import tensorflow as tf 
import numpy as np
from models.model_parent_class import ModelsParentClass

class ModelBasicCNN(ModelsParentClass):

    def __init__(self,learning_rate,training,nb_classes,num_epochs,train_iters,valid_iters,batch_size,train_data,valid_data):
        #mode defines whether the model is training or infering
        self.training = training
        self.learning_rate = learning_rate
        self.nb_classes = nb_classes
        self.num_epochs = num_epochs
        self.train_iters = train_iters
        self.valid_iters = valid_iters
        self.batch_size = batch_size
        self.train_data = train_data
        self.valid_data = valid_data
        super()

    def forward_propogation(self,x):
        ip = tf.reshape(x,[-1,247,80,1])
        conv1 = tf.layers.conv2d(inputs=ip,filters=32,kernel_size=[3,3],padding="same")
        norm =  tf.layers.batch_normalization(conv1, training=self.training)
        activation = tf.nn.relu(norm)
        pool1 = tf.layers.max_pooling2d(inputs=activation,pool_size=[2,2],strides=2,name="pool1")
        
        conv2 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=[3,3],padding="same")
        norm =  tf.layers.batch_normalization(conv2, training=self.training)
        activation = tf.nn.relu(norm)
        pool2 = tf.layers.max_pooling2d(inputs=activation,pool_size=[2,2],strides=2,name="pool2")
        
        conv3 = tf.layers.conv2d(inputs=pool2,filters=32,kernel_size=[3,3],padding="same")
        norm =  tf.layers.batch_normalization(conv3, training=self.training)
        activation = tf.nn.relu(norm)
        pool3 = tf.layers.max_pooling2d(inputs=activation,pool_size=[2,2],strides=2,name="pool3")
        
        pool3_flat = tf.reshape(pool3,[-1,30*10*32])
        #dense = tf.layers.dense(inputs=pool4_flat,units=32,activation=tf.nn.relu)
        #dropout = tf.layers.dropout(inputs=dense,rate=0.4)
        logits = tf.layers.dense(inputs=pool3_flat,units=self.nb_classes,name='logits') 
        return logits

    def train(self):
        super().print_message() 
        #Creating a graph 
        graph = tf.Graph()

        with graph.as_default():
    
            #Create global step for savind models
            global_step = tf.train.get_or_create_global_step()

            #Create pipeline for loading training dataset
            filename = tf.placeholder(tf.string,shape=[None])
            dataset = tf.data.TFRecordDataset(filename)
            dataset = dataset.map(super().read_from_tfrecord)
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            spec,label = iterator.get_next()

            #Compute logits
            logits =self.forward_propogation(spec)

            #Compute cross entropy loss and record variable
            loss = tf.losses.sparse_softmax_cross_entropy(labels=label,logits=logits)
    
            tf.summary.scalar('loss',loss)


            #optimize weights
            decayed_lr = tf.train.exponential_decay(self.learning_rate,global_step,10000,0.95,staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(loss,global_step=global_step)

            #Calculate training_accuracy of model
            accuracy,confusion_matrix = super().compute_accuracy(logits,label,self.nb_classes)
            tf.summary.scalar('accuracy',accuracy)
            #Initialize variables and variable saver
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("logdir/binary/train")
            valid_writer = tf.summary.FileWriter("logdir/binary/valid")
    
        with tf.Session(graph=graph) as sess:

            #initialize model parameters
            sess.run([init])

            for epoch in range(self.num_epochs):
                sess.run([iterator.initializer],feed_dict={filename:[self.train_data]})
                t_c = 0 
                t_l = 0
                for train_iter in range(self.train_iters):
                    l,_,acc,summary = sess.run([loss,optimizer,accuracy,merged])
                    t_c+= np.squeeze(acc)
                    t_l+=np.squeeze(l)
                    train_writer.add_summary(summary,epoch*self.train_iters + train_iter)
                c = 0
                val_loss = 0
                count = 0
                cm = np.zeros([self.nb_classes,self.nb_classes])
                sess.run([iterator.initializer],feed_dict={filename:[self.valid_data]})
                for valid_iter in range(self.valid_iters):
                    acc,l,summary,confm = sess.run([accuracy,loss,merged,confusion_matrix])
                    c += np.squeeze(acc)
                    cm += np.squeeze(confm)
                    val_loss +=np.squeeze(l)
                    valid_writer.add_summary(summary,epoch*self.valid_iters + valid_iter)
                print("Validation accuracy:",float(c/self.valid_iters),"Validation loss:",float(val_loss/self.valid_iters),"Training accuracy:",float(t_c/self.train_iters),"Training loss:",float(t_l/self.train_iters))
            print(cm)
 
                #Save model for this epoch
                #saver.save(sess, './src/saved_model/nsynth_binary_classifier', global_step=global_step)

    def test_load_from_tfrecords(self):
        graph = tf.Graph()
        print("Train data path:",self.train_data)
        print("Valid data path:",self.valid_data)
        print("Batch size:",self.batch_size)
        with graph.as_default():
    
            #Create global step for savind models
            global_step = tf.train.get_or_create_global_step()

            #Create pipeline for loading training dataset
            filename = tf.placeholder(tf.string,shape=[None])
            dataset = tf.data.TFRecordDataset(filename)
            dataset = dataset.map(super().read_from_tfrecord)
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            spec,label = iterator.get_next()

        with tf.Session(graph=graph) as sess:
            
            sess.run([iterator.initializer],feed_dict={filename:[self.train_data]})
            s,l = sess.run([spec,label])
            print("Train labels:",l)
            sess.run([iterator.initializer],feed_dict={filename:[self.valid_data]})
            s,l = sess.run([spec,label])
            print("Valid labels:",l)
