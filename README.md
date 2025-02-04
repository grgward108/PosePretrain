# PosePretrain

## Overview

This project focuses on pretraining a transformer model on large datasets to enhance the spatial understanding and temporal understanding of human markers.
This module has 2 transformers, Temporal Transformer and Spatial Transformer. 

## Objectives

- Pretrain a transformer model on extensive datasets.
- Improve the model's ability to understand and predict human markers in various spatial contexts.

## Spatial Transformer
- Using a masking strategy to train the transformer to recover missing markers 
- Marker representation here is 143 markers.
- The objective is to train the Spatial Transformer to understand the spatial relationships between markers

for pretraining on multiple GPU, run
```
./run_pretrain.sh --script --spatial --exp_name {YOUR EXP NAME}
```

## Temporal Transformer
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


## Liftup Transformer
- Lift up 17 joints to 143 markers
- The objective is to use this on the Temporal Transformer output to recreate the 143 markers.  

for pretraining, run
```
python train_liftup_transformer.py --exp_name {YOUR EXP NAME}
```

for finetuning on the GRAB dataset, run
```
python finetune_liftup_transformer.py --exp_name {YOUR EXP NAME}
```
