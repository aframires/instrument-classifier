from __future__ import absolute_import, print_function, division

import tensorflow as tf

tf.enable_eager_execution()


import argparse




def _parse_and_lmel(example_proto):

    features = {
        "note_str": tf.FixedLenFeature([], dtype=tf.string),
        "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
        "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
        "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
        "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
        "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
        "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
    }

    print(example_proto)

    parsed_features = tf.parse_single_example(example_proto,features)

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.contrib.signal.stft(parsed_features["audio"], frame_length=1024, frame_step=256,
                                   fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
    #spec_feature = tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(log_mel_spectrograms,[-1])))
    #shape_feature = tf.train.Feature(int64_list=tf.train.int64_list(value=[247,80]))
    #example = tf.train.Example(features=tf.trainFeatures(
    #    feature={"note_str": parsed_features["note_str"], "pitch": parsed_features["pitch"], "velocity": parsed_features["velocity"], "spectogram": spec_feature, "spec_shape": shape_feature, "qualities": parsed_features["qualities"], "instrument_source": parsed_features["instrument_source"],  "instrument_family": parsed_features["instrument_family"]}
    #))
    #with tf.python_io.TFRecordWriter("F:\\Code\\Data\\nsynth-test-spec.tfrecord") as writer:
    #    writer.write(example.SerializeToString())
    return parsed_features["note_str"], parsed_features["pitch"], parsed_features["velocity"], tf.reshape(log_mel_spectrograms,[-1]), [247,80], parsed_features["qualities"], parsed_features["instrument_source"], parsed_features["instrument_family"]








def feature_extraction_fullnsynth(directory, write_directory):
    ##Nice command sets priority of the process. This is important if you are sharing GPU resources!
    # os.nice(20)

    ##Add path to data directory here!
    # directory = "/datasets/MTG/projects/NSynth/nsynth-valid/audio/"

    ##Write directory
    # write_directory = "/homedtic/aramires/NSynth/nsynth-valid-spec.tfrecord"
    graph = tf.Graph()
    with graph.as_default():
        #print(directory)
        #print(write_directory)
        dataset = tf.data.TFRecordDataset(directory)
        dataset = dataset.map(_parse_and_lmel)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
    with tf.python_io.TFRecordWriter(write_directory) as writer:
        with tf.Session(graph=graph) as sess:
            try:

                while True:



                        data_record = sess.run(next_element)
                        #print(data_record)
                        #print(len(data_record))
                        #print(type(data_record[0]))

                        #feature = {
                        #    "note_str": tf.Fixe
                        #    "pitch": tf.FixedLe
                        #    "velocity": tf.Fixe
                        #    "spec": tf.FixedLe
                        #    "specDim":
                        #    "qualities": tf.Fix
                        #    "instrument_source":
                        #    "instrument_family":


                        #}
                        note_str_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[0]))
                        pitch_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[1]))
                        velocity_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[2]))
                        spec_feature = tf.train.Feature(float_list= tf.train.FloatList(value=data_record[3]))
                        spec_dim_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[4]))
                        qualities_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[5]))
                        instrument_source_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[6]))
                        instrument_family_feature = tf.train.Feature(int64_list= tf.train.Int64List(value = data_record[7]))

                        example = tf.train.Example(features=tf.train.Features(
                            feature={'note_str':note_str_feature, 'pitch':pitch_feature, 'velocity': velocity_feature, 'spec':spec_feature,
                            'spec_dim':spec_dim_feature, 'qualities':qualities_feature, 'instrument_source':instrument_source_feature, 'instrument_family':instrument_family_feature}))

                        writer.write(example.SerializeToString())
            except tf.errors.OutOfRangeError:
                pass
            print("finished!")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds a spectogram field to the TFRecord")
    parser.add_argument("input", help="Input folder")
    parser.add_argument("output", help="Output TFRecord file")
    args = parser.parse_args()
    feature_extraction_fullnsynth(args.input, args.output)
