# Capybara Semantic Segmentation with SegFormer

## Overview
This project implements a lightweight SegFormer-based semantic segmentation pipeline for capybara image segmentation.

## Features
- VOC-style dataset loader
- Lightweight SegFormer-B0 model
- Training and validation pipeline

## Dataset
The dataset is organized in VOC format:

VOCdevkit/VOC2007/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/

## Training
python train.py --config configs/segformer_b0_capybara.yaml


## Results

## Demo


## Acknowledgements
This project is inspired by the open-source SegFormer implementation from bubbliiiing/segformer-pytorch and the original SegFormer paper.
