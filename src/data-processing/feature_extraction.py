from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np
import argparse
import os
from scipy.io import wavfile
import librosa


def feature_extraction_fullnsynth(directory, write_directory,n_mfcc, effect=None):
    ##Nice command sets priority of the process. This is important if you are sharing GPU resources!
    # os.nice(20)

    ##Add path to data directory here!
    # directory = "/datasets/MTG/projects/NSynth/nsynth-valid/audio/"

    ##Write directory
    # write_directory = "/homedtic/aramires/NSynth/nsynth-valid-spec.tfrecord"

    print(directory)
    print(write_directory)
    graph = tf.Graph()
    with graph.as_default():
        pcm = tf.placeholder(tf.float32, [None, None])

        # A 1024-point STFT with frames of 64 ms and 75% overlap.
        stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=256,
                                       fft_length=1024)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        lower_edge_hertz, upper_edge_hertz = 40.0, 7600.0
        num_mel_bins = n_mfcc
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
        print(log_mel_spectrograms.shape)


    # The word_list is the list of instrument families you are interested in
    word_list = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead',
                 'vocal']

    count = np.zeros(11)
    iter = 0
    with tf.python_io.TFRecordWriter(write_directory) as writer:
        with tf.Session(graph=graph) as sess:
            for files in os.listdir(directory):
                if (files.endswith(effect + ".wav") and any(word in files for word in word_list)):
                    if (files.find("bass") != -1):
                        label = 0
                        count[0] += 1
                    elif (files.find("brass") != -1):
                        label = 1
                        count[1] += 1
                    elif (files.find("flute") != -1):
                        label = 2
                        count[2] += 1
                    elif (files.find("guitar") != -1):
                        label = 3
                        count[3] += 1
                    elif (files.find("keyboard") != -1):
                        label = 4
                        count[4] += 1
                    elif (files.find("mallet") != -1):
                        label = 5
                        count[5] += 1
                    elif (files.find("organ") != -1):
                        label = 6
                        count[6] += 1
                    elif (files.find("reed") != -1):
                        label = 7
                        count[7] += 1
                    elif (files.find("string") != -1):
                        label = 8
                        count[8] += 1
                    elif (files.find("synth_lead") != -1):
                        label = 9
                        count[9] += 1
                    elif (files.find("vocal") != -1):
                        label = 10
                        count[10] += 1

                    # Read the audio file and normalize between -1 and 1
                    fs, data = wavfile.read(os.path.join(directory, files))
                    data = data[0:64000] #cut longer sounds
                    data = data / np.max(np.abs(data))
                    x = np.reshape(data, [1, -1])

                    # Compute log mel spectrogram of the audio
                    spec = sess.run([log_mel_spectrograms], feed_dict={pcm: x})
                    #print("spec:")
                    #print(len(spec))
                    # Prepare the data to store as a tfrecord file
                    spec = np.asarray(spec, np.float32)
                    spec = np.squeeze(spec)
                    spec_raw = spec.tostring()
                    shape = np.array(spec.shape, np.int32)
                    shape_raw = shape.tostring()
                    #label = np.array(label, np.int64)
                    label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    spec_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[spec_raw]))
                    shape_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape_raw]))
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'spec': spec_feature, 'shape': shape_feature, 'label': label_feature}))
                    writer.write(example.SerializeToString())
                    iter += 1
                    if (iter % 1000 == 0):
                        print(iter)
        print(count)
    return


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Adds a spectogram field to the TFRecord")
    #parser.add_argument("--input", help="Input folder")
    #parser.add_argument("--output", help="Output TFRecord file")
    #arser.add_argument("--n_mfcc", help="Number of MFCC", type=int)
    #args = parser.parse_args()
    #feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-bitcrusher-spec.tfrecord", 80,'bitcrusher')
    #feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-chorus-spec.tfrecord", 80,'chorus')
    #feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-delay-spec.tfrecord", 80,'delay')
    #feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-flanger-spec.tfrecord", 80,'flanger')
    #feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-reverb-spec.tfrecord", 80,'reverb')
    #feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-tube-spec.tfrecord", 80,'tube')
    feature_extraction_fullnsynth("F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\", "F:\\Code\\Data\\NSynth\\nsynth-train-pitch_shifting-spec.tfrecord", 80,'pitch_shifting')