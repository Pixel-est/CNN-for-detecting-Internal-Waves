# Automated Detection of Internal Waves using SAR

## Overview
This project develops a CNN-based framework to detect internal waves in Sentinel-1 SAR imagery and analyse seasonal variability.

## Methods
- Data acquisition via Sentinel-1 API
- Image tiling and preprocessing
- CNN classification (SqueezeNet-based)
- Time-series seasonal analysis

## Requirements
- Python 3.x
- TensorFlow / PyTorch
- MATLAB (for preprocessing scripts)

## Usage
-Set up coordinates and time scale for AOI in python and use sentinel1_api_v2.py to download.
-Import images into S-1 Folder and run IW_Monthly_Presence_Summary.m
## Author
Leo Searl – University of Plymouth Dissertation Project (2026)
