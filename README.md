# Project Setup

To set up the project on a Linux system, please execute the following commands:



```
git clone https://github.com/kheuer/xai_proj_m.git
cd xai_proj_m
pip install -r requirements.txt
curl -L -o pacs-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nickfratto/pacs-dataset
unzip pacs-dataset.zip -d pacs-dataset
rm -f pacs-dataset.zip 
```


## Dataset Preparation

To train or validate results from the Camelyon Dataset, please download it from [kaggle](https://www.kaggle.com/datasets/mahdibonab/camelyon17) and save it in the project's root directory structured as follows:

```
project
├── camelyon17
│   ├── data
│   │   └── camelyon17_v1.0
│   │       ├── metadata.csv
│   │       ├── patches
│   │       └── RELEASE_v1.0.txt
│   └── src
│       └── BigQuery_Helper
│           ├── bq_helper
│           ├── bq_helper.egg-info
│           ├── LICENSE
│           ├── README.md
│           ├── setup.py
│           └── version.py
...
```

> Note: The Dlow augmentation strategy requires proprietary NVIDIA drivers and CUDA support. Running the augmentation without CUDA will lead to issues.

## Finding the Optimal Hyperparameters

To find the optimal hyperparameters, execute the following command and follow the prompts:


```
python tuner.py
```
## Running a Single Experiment Repeatedly

To train a model with specific hyperparameter combinations and obtain the test loss, run:



```
python tuner.py
```

You will be prompted for hyperparameters. You can utilize values obtained from the tuner, create your own, or use the following defaults:


```
{"EPOCHS": 100, "PATIENCE": 15, "BATCH_SIZE": 32, "LEARNING_RATE": 0.001, "BETA_1": 0.9, "BETA_2": 0.999, "OPTIMIZER": "SGD", "SCHEDULER": "CosineAnnealingLR", "MOMENTUM": 0.53, "DAMPENING": 0.0145, "WEIGHT_DECAY": 0.0}
```


## Training the Model Repeatedly for Different Combinations of Hyperparameters

To replicate the previously obtained results, execute the following lines:

```
python train_for_comparision_study.py --dataset_name pacs
python train_for_comparision_study.py --dataset_name camelyon
python train_for_comparision_study.py --dataset_name camelyon_unbalanced
```
This procedure will initialize and train a ResNet model repeatedly, with total runs depending on the dataset: 192 runs for PACS, and 160 runs each for the Camelyon datasets. 

> Note: This process may take an extended period (up to several weeks depending on your machine) but is crash persistent. CUDA support is required for training on the PACS dataset.

## Validating Results

To validate results from the saved checkpoints, run:

```
python validate_results.py --dataset_name pacs
python validate_results.py --dataset_name camelyon
python validate_results.py --dataset_name camelyon_unbalanced
```