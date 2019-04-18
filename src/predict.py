
import tensorflow as tf
import argparse


def load_and_predict(model_path, test_set_path,test):
    model = tf.keras.models.load_model(model_path,custom_objects={'f1': f1,'precision': precision,'recall':recall})
    dataset = create_dataset(test_set_path)

    if test == 0:
        metrics = model.evaluate(dataset, steps=12678, verbose=0)  # 12678

    if test == 1:
        model.evaluate(dataset, steps=4096, verbose=0)
    print(model_path + '\tacc ' + str(metrics[1]))


def load_and_predict_with_dataset(model_path, dataset,test,filename):
    model = tf.keras.models.load_model(model_path)

    if test == 0:
        metrics = model.evaluate(dataset, steps=12678, verbose=2)  # 12678

    if test == 1:
        model.evaluate(dataset, steps=4096, verbose=2)

    print(model_path + '\tacc ' + str(metrics[1]))

    with open(filename + ".txt", "a") as myfile:
        myfile.write(model_path + '\tacc ' + str(metrics[1]) + '\tf1 ')


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function)  # num_parallel_calls=8
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the batchsize
    dataset = dataset.batch(50)
    return dataset


def _parse_function(example):
    tfrecord_features = tf.parse_single_example(example, features={'spec': tf.FixedLenFeature([], tf.string),
                                                                   'shape': tf.FixedLenFeature([], tf.string),
                                                                   'label': tf.FixedLenFeature([], tf.int64)},
                                                name='features')
    spec = tf.decode_raw(tfrecord_features['spec'], tf.float32)
    spec = tf.reshape(spec, [80, 247, 1])
    label = tfrecord_features['label']
    label = tf.one_hot(label, 11)
    return spec, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates the given model in a given dataset.')
    parser.add_argument("--model", dest="mdir", help="model directory")
    parser.add_argument("--dataset", dest="ddir", help="dataset directory")
    parser.add_argument("--testing", dest="testing", help="If the models is testing (1) or valid (0)")
    args = parser.parse_args()
    load_and_predict(args.mdir, args.ddir)