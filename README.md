# Overview

This repository is a research work by Lindvall Lab at Dana-Farber Cancer Institute on extracting present/current symptoms reported by the patients from their electronic health record (EHR). Symptoms are vital outcomes for cancer clinical trials, observational research, and population-level surveillance. We sought to develop, test, and externally validate a deep learning model to extract symptoms from unstructured clinical notes in the electronic health record (EHR).

# Project Pipeline

- Processing
  - [Process label-studio annotation output for model input](README.md#how-to-process-annotation-output-labeltext-for-model-input)
- Training
  - [Run the models](README.md#how-to-run-the-models)
  - [Hyperparameters optimization](README.md#how-to-optimize-the-hyperparameters)
  - [Upload the model output to annotation server](README.md#how-to-upload-the-model-output-to-annotation-server)
- Inference
  - [Use best model for predictions](README.md#use-best-performing-model-for-predictions)
  - [Copyright](README.md#copyright)


## How to process annotation output label/text for model input
- [Processing the label-studio output](processing/label_output.py)
```
python processing/label_output.py \
  --input {location of the label-studio output json files} \
  --label_config {configuration used to set up label-studio; xml file} \
  --label all OR --keep goals_or care
  --hpi \
  --stratified_split 0.3 \
  --test
```
- Without `--test` argument, data will be stratified split to train/valid 0.7/0.3
- With `--test` argument, data will be stratified split to train/valid/test 0.7/0.15/0.15
- It takes around 17s to load the spacy `en_core_sci_lg` model, please wait.

## Training
### Run the models
- Transformer model choices: 'bert', 'xlnet', 'roberta', 'xlm-roberta', 'camembert', 'distilbert', 'electra'
```
conda activate transformers
python ner.py \
  --dset {location of the data that has been converted to ConLL format} \
  --model_class electra \
  --pretrained_model google/electra-base-discriminator \
  --lr 6e-5 \
  --decay 0.02 \
  --warmups 500
```

### Optimize the hyperparameters
- Bayesian optimization with Gaussian processes
  - Please open the interactive plots (contour_plot, slice_plot, cv_plot, etc) in browser
  
```
python optimization.py \
  --model bert \
  --lr 1e-6 1e-4 \
  --decay 0.01 0.1 \
  --warmups 0 3000 \
  --eps 1e-9 1e-7
```
### Load model outputs back into server hosting label studio - for active learning
```
python processing/model_output.py \
  --model_output processing/output/symptoms_hpi_all/prediction_test.txt \
  --label_output_dir symptoms/storage/label-studio/project/completions/ \
  --label_config symptoms/storage/label-studio/project/config.xml
```

## Inference
Use raw csv files with a column containing clinical note - no need to convert into ConLL format.
```
python inference/run_and_predict.py -ipf {location of the input file} -opf {location of dummy output file} -cn {name of the column containing the clinical note}
```

## Copyright
All codes are modified from
- [Label-studio-transformers](https://github.com/heartexlabs/label-studio-transformers)
- [GraphIE](https://github.com/thomas0809/GraphIE/tree/master/word-level)
- [huggingface](https://github.com/huggingface/transformers)

# License

The GNU GPL v2 version of PathML is made available via Open Source licensing. 
The user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.

Commercial license options are available also.

# Contact

Questions? Comments? Suggestions? Get in touch!

[CHARLOTTA_LINDVALL@DFCI.HARVARD.EDU](mailto:CHARLOTTA_LINDVALL@DFCI.HARVARD.EDU)
