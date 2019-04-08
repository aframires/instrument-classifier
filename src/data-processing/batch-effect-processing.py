import argparse
import os
import librosa
import random
random.seed(0)


def apply_effect(input_dir, output_dir, effect, mrswatson):
    files = [file_i
             for file_i in os.listdir(input_dir)
             if file_i.endswith('.wav')]

    for file in files:

        if effect == 'delay':
            command = mrswatson + 'mrswatson.exe --plugin-root "C:\Program Files (x86)\Vstplugins" -p "TAL-Dub-2" -i ' + input_dir + file + ' -o ' + output_dir + os.path.splitext(file)[0] + 'delay.wav'
            print(command)
            os.system(command)

        if effect == 'bitcrusher':
            command = mrswatson + 'mrswatson.exe --plugin-root "C:\Program Files (x86)\Vstplugins" -p "TAL-Bitcrusher" -i ' + input_dir + file + ' -o ' + output_dir + \
                      os.path.splitext(file)[0] + 'bitcrusher.wav'
            print(command)
            os.system(command)

        if effect == 'chorus':
            command = mrswatson + 'mrswatson.exe --plugin-root "C:\Program Files (x86)\Vstplugins" -p "TAL-Chorus-LX" -i ' + input_dir + file + ' -o ' + output_dir + os.path.splitext(file)[0] + 'chorus.wav'
            print(command)
            os.system(command)

        if effect == 'flanger':
            command = mrswatson + 'mrswatson.exe --plugin-root "C:\Program Files (x86)\Vstplugins" -p "TAL-Flanger" -i ' + input_dir + file + ' -o ' + output_dir + os.path.splitext(file)[0] + 'flanger.wav'
            print(command)
            os.system(command)

        if effect == 'reverb':
            command = mrswatson + 'mrswatson.exe --plugin-root "C:\Program Files (x86)\Vstplugins" -p "TAL-Reverb-4" -i ' + input_dir + file + ' -o ' + output_dir + os.path.splitext(file)[0] + 'reverb.wav'
            print(command)
            os.system(command)

        if effect == 'tube':
            command = mrswatson + 'mrswatson.exe --plugin-root "C:\Program Files (x86)\Vstplugins" -p "TAL-Tube" -i ' + input_dir + file + ' -o ' + output_dir + os.path.splitext(file)[0] + 'tube.wav'
            print(command)
            os.system(command)

        if effect == 'pitch_shifting':
            y, sr = librosa.core.load(input_dir + file, sr=None)
            n_divisions_step = 6
            n_steps = random.randint(1, n_divisions_step-1)
            y_shifted = librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12 * n_divisions_step)
            librosa.output.write_wav(output_dir + os.path.splitext(file)[0] + 'pitch_shifting.wav', y_shifted, sr, norm=False)
    return

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Applies an effect to all the files in one directory and saves the result to an output directory")
    #parser.add_argument("input", help="Input directory")
    #parser.add_argument("output", help="Output directory")
    #parser.add_argument("effect", help="Effect to be applied")
    #parser.add_argument("mrswatson", help="MrsWatson Path")
    #args = parser.parse_args()

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\\main\\Release\\"
    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    effect = 'pitch_shifting'
    apply_effect(input, output, effect, mrswatson)

"""
    effect = 'delay'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    #apply_effect(input, output, effect, mrswatson)


    effect = 'chorus'


    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    #apply_effect(input, output, effect, mrswatson)


    effect = 'bitcrusher'


    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    #apply_effect(input, output, effect, mrswatson)







    effect = 'reverb'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson)

    effect = 'tube'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson)

    effect = 'flanger'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson)
    """

