from predict import load_and_predict_with_dataset, create_dataset
import os
import argparse


#load_and_predict(args.mdir,args.ddir,args.testing)


#test sets
def predit_for_directory(models_directory,datasets_directory,training,effectm,effectv):
    print('entrei')
    for dataset_path in os.listdir(datasets_directory):
        dataset = create_dataset(datasets_directory+ "nsynth-valid" + effectv + "-spec.tfrecord")
        print(dataset_path)
        for model in os.listdir(models_directory):
            if  effectm in model:
                load_and_predict_with_dataset(models_directory + model, dataset,training, effectm + effectv)



if __name__ == "__main__":
    print('im in')
    parser = argparse.ArgumentParser(description='Evaluates the given model in a given dataset.')
    parser.add_argument("--exp", dest="exp", help="model directory")
    args = parser.parse_args()
    if args.exp == '1':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all-valid_','-pitch_shifting')
    if args.exp == '2':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all-valid_','-bitcrusher')
    if args.exp == '3':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all-valid_','-chorus')
    if args.exp == '4':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all-valid_','-delay')
    if args.exp == '5':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all_','-pitch_shifting')
    if args.exp == '6':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all_','-bitcrusher')
    if args.exp == '7':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all_','-chorus')
    if args.exp == '8':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             '_all_','-delay')
    if args.exp == '9':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             'tube','-tube')
    if args.exp == '10':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             'tube',"")
    if args.exp == '11':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             'pitch_shifting','-pitch_shifting')
    if args.exp == '12':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             'bitcrusher','-bitcrusher')
    if args.exp == '13':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             'chorus','-chorus')
    if args.exp == '14':
        predit_for_directory("/homedtic/aramires/instrument-classifier/instrument-classifier/src/trained_models/val/",
                             "/homedtic/aramires/NSynth/final/valid/", 0,
                             'delay','-delay')

    #predit_for_directory("C:\\experiment\\src\\trained_models\\val\\","C:\\experiment\\final\\valid\\",0)
    #predit_for_directory("C:\\experiment\\src\\trained_models\\val\\","C:\\experiment\\final\\test\\",1)
