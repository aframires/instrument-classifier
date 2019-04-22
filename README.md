# instrument-classifier
Framework for automatic instrument classification using the NSynth dataset and augmenting this dataset with audio effects. The repository contains code for creating of a TFRecords file with the spectograms and labels from the NSynth dataset, augmenting this dataset with VST plugins and training and evaluating an [instrument classification model](https://github.com/Veleslavia/EUSIPCO2017) on the augmented datasets.



## Data Pre-processing
We use the NSynth dataset, which is available [here](https://magenta.tensorflow.org/datasets/nsynth). For the data augmentation, the following plugins were used:

* [TAL-Bitcrusher](https://tal-software.com/products/tal-effects)
* [TAL-Tube](https://tal-software.com/products/tal-effects)
* [TAL-Reverb-4](https://tal-software.com/products/tal-reverb-4)
* [TAL-Chorus-LX](https://tal-software.com/products/tal-chorus-lx)
* [TAL-Dub-2](https://tal-software.com/products/tal-dub}}
* [TAL-Flanger](https://tal-software.com/products/tal-effects)
* [Camel Audio's CamelCrusher](https://www.kvraudio.com/product/camelcrusher-)
* [Shattered Glass Audio's Ace](http://www.shatteredglassaudio.com/product/103)
* [OrilRiver](https://www.kvraudio.com/product/orilriver-by-denis-tihanov)
* [Blue Cat's Chorus](https://www.bluecataudio.com/Products/Product_Chorus)
* [++delay\footnote](http://www.soundhack.com/freeware/)
* [Blue Cat's Flanger](https://www.bluecataudio.com/Products/Product_Flanger)

To apply the audio effects to wav files, [MrsWatson](https://github.com/teragonaudio/MrsWatson) should be installed.

After the Nsynth dataset has been downloaded and the effects and MrsWatson installed, `data-processing/batch-effect-processing.py` should be adapted to run on the desired directories. When this proccess finishes, the paths in `data-processing/feature_extraction.py` should be modified to the correct directories and, then, running this script will create TFRecord dataset files for each audio effect.

## Training and evaluation

To train the model, the `main.py` function should be called with the appropriate flags. The path to the training and validation sets should be set and the "effect" flag should be set to the desired effect: bitcrusher, chorus, delay, flanger, reverb, tube, pitch_shifting or none, if no effect is desired.

The evaluation of each model on a dataset can be performed by running `predict.py` with the apropriate flags.
