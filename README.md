# MNIST Digit Classification (Portfolio Project 01)

## Introduction
This repository contains a PyTorch implementation of a lightweight convolutional neural network (CNN) trained to recognize handwritten digits from the MNIST dataset. It demonstrates a complete end‑to‑end workflow — from data loading and model design to training, evaluation, and visualization — and serves as a showcase project for building a computer vision portfolio.

## File Structure
├── checkpoints/ # Saved model checkpoints (*.pth) 
├── visualizations/ # Generated plots (PNG) 
├── data.py # DataLoader definitions 
├── model.py # CNN architecture definition 
├── train.py # Training loop + checkpoint saving 
├── eval.py # Load checkpoint → compute test metrics 
├── visualize.py # Dataset & prediction visualizations ├── requirements.txt # dependencies 
└── Portfolio Project 01.pdf # Detailed project report

## 🚀 Quickstart

### Install dependencies

```bash
pip3 install -r requirements.txt
```

### Model Training
```bash
python3 train.py
```

### Model Evaluation
```bash
python3 eval.py
```

### Generate Visualizations (Optional)
```bash
python3 visualize.py
```