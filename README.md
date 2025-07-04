# Setup
To setup the project run these commands on a linux system:

```
git clone https://github.com/kheuer/xai_proj_m.git
cd xai_proj_m
pip install -r requirements.txt
curl -L -o pacs-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nickfratto/pacs-dataset
unzip pacs-dataset.zip -d pacs-dataset
rm -f pacs-dataset.zip 
```

To run this using Google Colab, execute:
```
!git clone https://github.com/kheuer/xai_proj_m.git
%cd xai_proj_m
!pip install -r requirements.txt
!curl -L -o pacs-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nickfratto/pacs-dataset
!unzip pacs-dataset.zip -d pacs-dataset
!rm -f pacs-dataset.zip 
```


# How to proceed

Resnet 18
Pretrained = False
Augmentations = True

tune for all 4 hyperparam combinations and compare losses

use new dataset