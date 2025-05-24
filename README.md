# Implementation of A Unified Transformer-Based Framework with Pretraining For Whole Body Grasping Motion Generation

## Environment


```
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install human-body-prior
pip install git+https://github.com/vchoutas/smplx.git
pip install git+https://github.com/otaheri/chamfer_distance.git
pip install open3d==14.2
pip install pytorch-ignite
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
```
This environment is built upon the SAGA environment. If you find any trouble, try to build their environment first. 


## Overview

This project focuses on pretraining a transformer model on large datasets to enhance the spatial understanding and temporal understanding of human markers.
This module has 2 transformers, Temporal Transformer and Spatial Transformer. 

## Objectives

- Pretrain a transformer model on extensive datasets.
- Improve the model's ability to understand and predict human markers in various spatial contexts.

## Generalized Spatial Transformer
- Using a masking strategy to train the transformer to recover missing markers 
- Marker representation here is 143 markers.
- The objective is to train the Spatial Transformer to understand the spatial relationships between markers

for pretraining on multiple GPU, run
```
./run_pretrain.sh --script --spatial --exp_name {YOUR EXP NAME}
```

## Generalized Spatio-Temporal Transformer
- Using a masking strategy to train transformer to recover missing frames 
- Marker representation here is 22 joints
- The objective is to train the Temporal Transformer to understand the temporal relationships between markers

for pretraining on multiple GPU, run 
```
./run_pretrain.sh --script --temporal --exp_name {YOUR EXP NAME}
```

for finetuning on to the GRAB dataset, run
```
python finetune_temporal_transformer_with_pelvis.py --exp_name {YOUR EXP NAME}
```


## Generalized Liftup Transformer
- Lift up 22 joints to 143 markers
- The objective is to use this on the Temporal Transformer output to recreate the 143 markers.  

for pretraining, run
```
python train_liftup_transformer.py --exp_name {YOUR EXP NAME}
```

for finetuning on the GRAB dataset, run
```
python finetune_liftup_transformer.py --exp_name {YOUR EXP NAME}
```

## Weights
Download here for the weights and save it inside folder pretrained_models https://drive.google.com/drive/folders/15RkiTSj-09yszFDOZ3MX88VKvZc-TvSw?usp=sharing
