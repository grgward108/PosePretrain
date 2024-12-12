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

## Temporal Transformer
- Using a masking strategy to train transformer to recover missing frames 
- Marker representation here is 17 joints. (Following Human3.6M)
- The objective is to train the Temporal Transformer to understand the temporal relationships between markers

## Liftup Transformer
- Lift up 17 joints to 143 markers
- The objective is to use this on the Temporal Transformer output to recreate the 143 markers.  
