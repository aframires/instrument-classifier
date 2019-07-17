import argparse
import os
import librosa
import random

random.seed(0)

VST_PATH = "C:\\Program Files (x86)\\VSTPlugins"

def apply_effect(input_dir, output_dir, effect, mrswatson, set):
    files = [file_i
             for file_i in os.listdir(input_dir)
             if file_i.endswith('.wav')]

    if set == 'train':

        for file in files:

            if effect == 'delay':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "TAL-Dub-2" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'delay.wav'
                print(command)
                os.system(command)

            if effect == 'bitcrusher':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "TAL-Bitcrusher" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'bitcrusher.wav'
                print(command)
                os.system(command)

            if effect == 'chorus':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "TAL-Chorus-LX" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'chorus.wav'
                print(command)
                os.system(command)

            if effect == 'flanger':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "TAL-Flanger" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'flanger.wav'
                print(command)
                os.system(command)

            if effect == 'reverb':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "TAL-Reverb-4" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'reverb.wav'
                print(command)
                os.system(command)

            if effect == 'tube':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "TAL-Tube" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'tube.wav'
                print(command)
                os.system(command)

            if effect == 'pitch_shifting':
                y, sr = librosa.core.load(input_dir + file, sr=None)
                n_divisions_step = 6
                n_steps = random.randint(1, n_divisions_step - 1)
                y_shifted = librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12 * n_divisions_step)
                librosa.output.write_wav(output_dir + os.path.splitext(file)[0] + 'pitch_shifting.wav', y_shifted, sr,
                                         norm=False)
        return

    if set == 'valid' or set == 'test':

        for file in files:

            if effect == 'delay':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "++delay" --parameter 0,0.3 --parameter 1,0.25 -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'delay.wav'
                print(command)
                os.system(command)

            if effect == 'bitcrusher':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "CamelCrusher" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'bitcrusher.wav'
                print(command)
                os.system(command)

            if effect == 'chorus':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "BC Chorus 4 VST(Mono)" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'chorus.wav'
                print(command)
                os.system(command)

            if effect == 'flanger':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "BC Flanger 3 VST(Mono)" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'flanger.wav'
                print(command)
                os.system(command)

            if effect == 'reverb':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "OrilRiver" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'reverb.wav'
                print(command)
                os.system(command)

            if effect == 'tube':
                command = mrswatson + 'mrswatson.exe --plugin-root ' + VST_PATH + ' -p "ace" -i ' + input_dir + file + ' -o ' + output_dir + \
                          os.path.splitext(file)[0] + 'tube.wav'
                print(command)
                os.system(command)

            if effect == 'pitch_shifting':
                y, sr = librosa.core.load(input_dir + file, sr=None)
                n_divisions_step = 6
                n_steps = random.randint(1, n_divisions_step - 1)
                y_shifted = librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12 * n_divisions_step)
                librosa.output.write_wav(output_dir + os.path.splitext(file)[0] + 'pitch_shifting.wav', y_shifted, sr,
                                         norm=False)

        return

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Applies an effect to all the files in one directory and saves the result to an output directory")
    #parser.add_argument("input", help="Input directory")
    #parser.add_argument("output", help="Output directory")
    #parser.add_argument("effect", help="Effect to be applied")
    #parser.add_argument("mrswatson", help="MrsWatson Path")
    #args = parser.parse_args()

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\\main\\Release\\"

    effect = 'pitch_shifting'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson)

    effect = 'delay'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'chorus'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'bitcrusher'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'reverb'

    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'tube'

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\\main\\Release\\"
    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'flanger'

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\main\\Release\\"
    input = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-train\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    ####################################

    effect = 'delay'

    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')


    effect = 'chorus'

    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')


    effect = 'bitcrusher'

    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')


    effect = 'reverb'

    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'tube'

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\\main\\Release\\"
    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')


    effect = 'flanger'

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\main\\Release\\"
    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    effect = 'pitch_shifting'

    mrswatson = "F:\\Code\\Research\\MrsWatson\\build\main\\Release\\"
    input = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-valid\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'valid')

    #####################################

    effect = 'delay'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'chorus'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'bitcrusher'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'reverb'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'tube'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'flanger'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'flanger'

    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')

    effect = 'pitch_shifting'
    input = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\"
    output = "F:\\Code\\Data\\NSynth\\nsynth-test\\audio\\processed\\"
    apply_effect(input, output, effect, mrswatson, 'test')
