
import tensorflow as tf
import argparse



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


"""
def mean_pred(y_true, y_pred):
    global YTRUES
    YTRUES = y_true
    global YPREDS
    YPREDS = y_pred

    return tf.keras.mean(y_pred)
"""

def load_and_predict(model_path, test_set_path,test):
    model = tf.keras.models.load_model(model_path,custom_objects={'f1': f1,'precision': precision,'recall':recall})
    dataset = create_dataset(test_set_path)

    if test == 0:
        metrics = model.evaluate(dataset, steps=12678, verbose=0)  # 12678

    if test == 1:
        model.evaluate(dataset, steps=4096, verbose=0)


    print(model_path + '\tacc ' + str(metrics[1]) + '\tf1 ' + str(metrics[2]) + '\tprecision ' + str(metrics[3]) + '\trecall ' + str(metrics[4]))

def load_and_predict_with_dataset(model_path, dataset,test,filename):
    model = tf.keras.models.load_model(model_path,custom_objects={'f1': f1,'precision': precision,'recall':recall})

    if test==0:
        metrics = model.evaluate(dataset, steps=12678, verbose=2)  # 12678

    if test == 1:
        model.evaluate(dataset, steps=4096, verbose=2)


    print(model_path + '\tacc ' + str(metrics[1]) + '\tf1 ' + str(metrics[2]) + '\tprecision ' + str(metrics[3]) + '\trecall ' + str(metrics[4]))

    with open(filename + ".txt", "a") as myfile:
        myfile.write(model_path + '\tacc ' + str(metrics[1]) + '\tf1 ' + str(metrics[2]) + '\tprecision ' + str(metrics[3]) + '\trecall ' + str(metrics[4]))


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function)  # num_parallel_calls=8
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the batchsize
    dataset = dataset.batch(50)
    # Create an iterator
    #iterator = dataset.make_one_shot_iterator()
    # Create your tf representation of the iterator
    #sound, label = iterator.get_next()

    #return sound, label
    return dataset

def _parse_function(example):
    tfrecord_features = tf.parse_single_example(example, features={'spec': tf.FixedLenFeature([], tf.string),
                                                                       'shape': tf.FixedLenFeature([], tf.string),
                                                                       'label': tf.FixedLenFeature([], tf.int64)},
                                                                        name='features')
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    spec = tf.decode_raw(tfrecord_features['spec'], tf.float32)
    spec = tf.reshape(spec, [80,247,1])
    label = tfrecord_features['label']
    label = tf.one_hot(label, 11)
    return spec, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates the given model in a given dataset.')
    parser.add_argument("--model", dest="mdir", help="model directory")
    parser.add_argument("--dataset", dest="ddir", help="dataset directory")
    parser.add_argument("--testing", dest="testing", help="If the models is testing (1) or valid (0)")
    args = parser.parse_args()
    load_and_predict(args.mdir,args.ddir)