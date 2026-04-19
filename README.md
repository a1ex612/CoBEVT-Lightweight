# Unstructured L1 Pruning of CoBEVT
This repository contains the code for reproducing the pruning experiments in the thesis: *Unstructured L1 Pruning of CoBEVT for Lightweight Cooperative BEV Perception*.

## Prerequisites
- Python 3.8+
- PyTorch 1.13+
- CUDA 11.7+
- `thop` (for MACs calculation)
- The official [CoBEVT](https://github.com/DerrickXuNu/CoBEVT) codebase and OPV2V dataset

## Installation
1. Clone the official CoBEVT repository and set up the environment following their instructions.
2. Replace `opencood/tools/inference_camera.py` with the one in this repo.
3. Place `run_inference.py` in the CoBEVT root directory.

## Dataset Preparation
Download the OPV2V dataset following the official CoBEVT instructions and place it in the `opv2v/dataset/` directory.

## Pre-trained Model Preparation
Download the pre-trained CoBEVT models (`cobevt` and `cobevt_static`) from the official CoBEVT repository and place them in `opencood/logs/`.

## Running the Experiments
```bash
python run_inference.py
