# Clinical XLNet

This repo hosts pretraining and finetuning weights and relevant scripts for Clinical XLNet.

## Requirements

```
torch
argparse
copy
tqdm
matplotlib
numpy
pandas
time
sklearn
```

## Clinical XLNet Weights

To download pretrained Clinical XLNet, click the following links: [This]() only uses Nursing Notes to pretrain and [this]() uses the discharge summary to pretrain. [This]() is the finetuned weights for PMV task, and [this]() is the finetuned weights for Mortality task. 

## PMV and Mortality Prediction using Clinical XLNet

Below list the sample scripts for running prediction. You can also simply modify the label to do your own downstream prediction task.

### Using Finetuned weights for Mortality or PMV Prediction
```
python train.py \
  --data_dir DATA_FILE\
  --config_path CONFIG\
  --model_path MORTALITY/PMV_MODEL_PATH \
  --save_meta_finetune_path SAVE_PATH \
  --prediction_label Mortality/PMV \
  --Batch_Size_Meta 4 \
  --Learning_Rate_Meta 1e-5 \
  --Training_Epoch_Meta 4 \
  --Batch_Size_Finetune 128 \
  --Learning_Rate_Finetune 2e-5 \
  --Training_Epoch_Finetune 30 \
  --saving_notes_embed_batch_size 32 \
  --skip_meta_finetuned 
```

### Training your own mortality or PMV prediction model from pretraining ClinicalXLNet
```
python train.py \
  --data_dir DATA_FILE\
  --config_path CONFIG\
  --model_path PRETRAIN_MODEL_PATH \
  --save_meta_finetune_path SAVE_PATH \
  --prediction_label Mortality/PMV \
  --Batch_Size_Meta 4 \
  --Learning_Rate_Meta 1e-5 \
  --Training_Epoch_Meta 4 \
  --Batch_Size_Finetune 128 \
  --Learning_Rate_Finetune 2e-5 \
  --Training_Epoch_Finetune 30 \
  --saving_notes_embed_batch_size 32 
```

It will use the train.csv, val.csv, and test.csv from the (DATA_FILE) folder.

The results of AUROC and AUPRC will be printed out.

## Datasets

We use [MIMIC-III](https://mimic.physionet.org/about/mimic/). Please fufill the CITI training program in order to use it. To use your own notes dataset, further pretraining is recommended.

File system expected:

```
-data
   -train.csv
   -val.csv
   -test.csv
```

## Pretraining your own Clinical XLNet

We provide a [notebook](pretraining/pretrain-xlnet.ipynb) tutorial to pretrain your own Clinical XLNet.

## Preprocessing and cohort curation

We provide notebook for preprocessing clinical notes and curate the PMV cohort on MIMIC-III. It consists of two parts, [R script](cohort_curation/HST953_FALL2019_Cohort_Selection27Sep19.Rmd) generates the general mechanical ventilation cohort and this [notebook](cohort_curation/MechVent_Preprocessing.ipynb) generates the specific cohort, see papers for detailed cohort curation process.

Data file is expected to have column "TEXT", "HADM_ID" and "Label" (Note chunks, Admission ID, Label).

## Contact

Please contact kexinhuang@hsph.harvard.edu for help or submit an issue. 

## Citation

Please cite [arxiv]():
```
@article{clinicalxlnet,
author = {Kexin Huang and Abhishek Singh and Sitong Chen and Edward Moseley and Chin-ying Deng and Naomi George and Charlotta Lindvall},
title = {Clinical XLNet: Modeling Sequential Clinical Notes and Predicting Prolonged Mechanical Ventilation},
year = {2019},
journal = {arXiv:},
}

```




