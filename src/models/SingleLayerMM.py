import tensorflow as tf


from models.model_parent_class import ModelsParentClass

class ValCallback(tf.keras.callbacks.Callback):

   def __init__(self, val_sets):


       self.val_data = val_sets



   def on_epoch_end(self, epoch, logs={}):

        for dataset in self.val_data:

            metrics = self.model.evaluate(dataset['data'],steps=634,verbose=0)
            pred = self.model.predict(dataset['data'],steps=634,verbose=0)
            pred = numpy.argmax(pred, axis=1)
            true = numpy.loadtxt('labels.txt', dtype=int)
            macro_p, macro_r, macro_f, nada = precision_recall_fscore_support(true, pred, average='macro',
                                                                              labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
            micro_p, micro_r, micro_f, nada = precision_recall_fscore_support(true, pred, average='micro',
                                                                              labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10])

            print('Dataset ' + dataset['name'] + '\tEpoch ' + str(epoch) + '\t MaP ' + str(macro_p) + '\t MaR ' + str(macro_r) + '\t MaF ' + str(macro_f))
            print('Dataset ' + dataset['name'] + '\tEpoch ' + str(epoch) + '\t MiP ' + str(micro_p) + '\t MiR ' + str(micro_r) + '\t MiF ' + str(micro_f))

            for metric in range(len(metrics)):
                print('Dataset ' + dataset['name'] + '\tEpoch ' + str(epoch) + '\t' + self.model.metrics_names[metric] + ' ' + str(metrics[metric]))



class SingleLayerMM(ModelsParentClass):

    N_MEL_BANDS = 96
    SEGMENT_DUR = 247
    SHUFFLE_BUFFER = 1000
    EARLY_STOPPING_EPOCH = 15
    init_lr = 0.001

    set_of_effects = set(['bitcrusher','chorus','delay','flanger','reverb','tube','pitch_shifting'])

    def __init__(self, learning_rate, training, nb_classes, num_epochs, train_iters, valid_iters, batch_size,
                 train_data, valid_data,datasets):

        self.training = training
        self.learning_rate = learning_rate
        self.nb_classes = nb_classes
        self.num_epochs = num_epochs
        self.train_iters = train_iters
        self.valid_iters = valid_iters
        self.batch_size = batch_size
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = tf.keras.optimizers.Adam(lr=self.init_lr)
        self.datasets = datasets
        super()

    def vertical_filter_model(self):
        input_shape = (self.N_MEL_BANDS, self.SEGMENT_DUR, 1)
        channel_axis = 3
        melgram_input = tf.keras.layers.Input(shape=input_shape)

        m_sizes = [50, 70]
        n_sizes = [1, 3, 5]
        n_filters = [128, 64, 32]
        maxpool_const = 4
        maxpool_size = int(self.SEGMENT_DUR / maxpool_const)
        layers = list()

        for m_i in m_sizes:
            for i, n_i in enumerate(n_sizes):
                x = tf.keras.layers.Conv2D(n_filters[i], [m_i, n_i],
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                  name=str(n_i) + '_' + str(m_i) + '_' + 'conv')(melgram_input)
                x = tf.keras.layers.BatchNormalization(axis=channel_axis, name=str(n_i) + '_' + str(m_i) + '_' + 'bn')(x)
                x = tf.keras.layers.ELU()(x)
                x = tf.keras.layers.MaxPool2D(pool_size=(self.N_MEL_BANDS, maxpool_size), name=str(n_i) + '_' + str(m_i) + '_' + 'pool')(
                    x)
                x = tf.keras.layers.Flatten(name=str(n_i) + '_' + str(m_i) + '_' + 'flatten')(x)
                layers.append(x)
        x = tf.keras.layers.concatenate(layers)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.nb_classes, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-5), activation='softmax', name='prediction')(x)
        model = tf.keras.Model(melgram_input, x)
        return model

    def train(self):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.EARLY_STOPPING_EPOCH)
        train_dataset = self.create_dataset(self.train_data)
        val_dataset = self.create_dataset(self.valid_data)
        dataset_dict = [{'name': 'none', 'data': val_dataset}]

        if self.datasets == 'all':
            index_of_train = self.train_data.find("-spec")
            index_of_valid = self.valid_data.find("-spec")
            self.train_iters = self.train_iters * (1 + len(self.set_of_effects))
            for effect in self.set_of_effects:
                extended_data_path = self.train_data[:index_of_train] + '-' + effect + self.train_data[index_of_train:]
                extended_dataset = self.create_dataset(extended_data_path)
                train_dataset.concatenate(extended_dataset)
                train_dataset.shuffle(self.SHUFFLE_BUFFER)
            
            for effect in self.set_of_effects:

                extended_valid_path = self.valid_data[:index_of_valid] + '-' + effect + self.valid_data[
                                                                                               index_of_valid:]
                valid_effect_dataset = self.create_dataset(extended_valid_path)
                dataset_to_add = {'name': effect, 'data': valid_effect_dataset}
                dataset_dict.append(dataset_to_add)
            print('Added all effects')

        elif self.datasets == 'all-valid':
            index_of_valid = self.valid_data.find("-spec")
            #self.valid_iters = self.valid_iters * (1 + len(self.set_of_effects))
            dataset_dict = [{'name': 'none', 'data': val_dataset}]
            for effect in self.set_of_effects:

                extended_valid_path = self.valid_data[:index_of_valid] + '-' + effect + self.valid_data[
                                                                                               index_of_valid:]
                valid_effect_dataset = self.create_dataset(extended_valid_path)
                dataset_to_add = {'name': effect, 'data': valid_effect_dataset}
                dataset_dict.append(dataset_to_add)

            print('Added all effects')

        elif self.datasets in self.set_of_effects:
            index_of_train = self.train_data.find("-spec")
            extended_data_path = self.train_data[:index_of_train] + '-' + self.datasets + self.train_data[index_of_train:]
            extended_dataset = self.create_dataset(extended_data_path)
            train_dataset.concatenate(extended_dataset)
            train_dataset.shuffle(self.SHUFFLE_BUFFER)
            self.train_iters = self.train_iters * 2

            index_of_valid = self.valid_data.find("-spec")
            extended_valid_path = self.valid_data[:index_of_valid] + '-' + self.datasets + self.valid_data[index_of_valid:]
            valid_effect_dataset = self.create_dataset(extended_valid_path)
            dataset_dict = [{'name':'none','data':val_dataset},{'name':self.datasets,'data':valid_effect_dataset}]
            #self.valid_iters = self.valid_iters * 2
            print('Added ' + self.datasets)

        model = self.vertical_filter_model()
        save_clb = tf.keras.callbacks.ModelCheckpoint(
            './SingleLayer_' + self.datasets + "_epoch.{epoch:02d}-val_loss.{val_loss:.3f}",
            monitor='val_loss',
            save_best_only=False)
        validation_saver = ValCallback(dataset_dict)
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )

        model.fit(train_dataset,
                  steps_per_epoch=self.train_iters,
                  epochs=self.num_epochs,
                  verbose=2,
                  callbacks=[save_clb, early_stopping, validation_saver],
                  validation_data=val_dataset,
                  validation_steps=self.valid_iters)
        return

    def create_dataset(self, filepath):
        # This works with arrays as well
        dataset = tf.data.TFRecordDataset(filepath)
        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function)  # num_parallel_calls=8
        # This dataset will go on forever
        dataset = dataset.repeat()
        # Set the number of datapoints you want to load and shuffle
        dataset = dataset.shuffle(self.SHUFFLE_BUFFER)
        # Set the batchsize
        dataset = dataset.batch(self.batch_size)

        return dataset

    def _parse_function(self, example):
        tfrecord_features = tf.parse_single_example(example, features={'spec': tf.FixedLenFeature([], tf.string),
                                                                           'shape': tf.FixedLenFeature([], tf.string),
                                                                           'label': tf.FixedLenFeature([], tf.int64)},
                                                                            name='features')
        spec = tf.decode_raw(tfrecord_features['spec'], tf.float32)
        spec = tf.reshape(spec, [self.N_MEL_BANDS,self.SEGMENT_DUR,1])
        label = tfrecord_features['label']
        label = tf.one_hot(label, self.nb_classes)
        return spec, label

