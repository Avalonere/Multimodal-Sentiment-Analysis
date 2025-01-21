# Multimodal Sentiment Analysis

This is the repository for the final project of the course "Contemporary Artificial Intelligence" at DaSE, ECNU. The
project is about multimodal sentiment analysis on the given dataset. The project is implemented in Python3 and PyTorch.

## Setup

To run the project, you need to install the required packages. You can install the required packages by running the
following command:

```bash
pip install -r requirements.txt
```

Dependencies include:

- `chardet`
- `matplotlib`
- `numpy`
- `openai_clip`
- `opencv_python`
- `pandas`
- `Pillow`
- `PyYAML`
- `scikit_learn`
- `seaborn`
- `torch`
- `torchvision`
- `tqdm`
- `umap_learn`

## Repository Structure

The repository is structured as follows:

```
.
├── checkpoints # Saved models, not packed due to size
│
├── data # Dataset # Dataset files, not packed due to size
│
├── experiments # Experiment logs
│
├── logs # Training logs
│
├── plots # Plots
│
├── vis # Visualization results
│
├── config.yaml # Configuration file
│
├── dataset.py # Dataset class
│
├── evaluate.py # Evaluation function and test set output
│
├── ensemble.py # Ensemble test set outputs
│
├── experiment.bat/sh # Experiment script
│
├── model.py # Model class
│
├── plot.py # Plot training log
│
├── predictions.txt # Test set predictions
│
├── preprocess.py # Preprocess dataset
│
├── README.md # This file
│
├── requirements.txt # Required packages
│
├── train.py # Training script
│
└── visualize.py # Visualization script
```

The `data` folder should be like:

```
    data
    ├── data # Data directory, containing images and texts
    │── train.txt
    │── test_without_label.txt
```

## Running Pipeline

The pipeline consists of the following steps:

1. Preprocess dataset:

```bash
python preprocess.py
```

This script unifies the dataset `txt` files to **UTF-8** encoding and saves the processed dataset to `data` directory.

2. Train model:

```bash
python train.py
```

This script trains the model on the training set and evaluates the model on the validation set. The training logs are
saved to `logs` directory and the best model is saved to `checkpoints` directory.

For configuration, you can modify the `config.yaml` file. The configuration file includes the following important
parameters:

```yaml
data:
  seed: 42 # Random seed for reproducibility
  imbalance_method: "weighted" # Method to handle class imbalance, "none", "weighted" or "sample"; "weighted" slightly amplifies the loss of minority classes, while "sample" oversamples minority classes 

training:
  batch_size: 64
  num_epochs: 10 # Maxium number of epochs to train
  learning_rate: 1e-4
  weight_decay: 1e-2
  clip_model: "ViT-L/14@336px" # CLIP model to use
  fusion: "concat" # Fusion method, "concat", "weighted", "attention" or "attention_alt"; neglected for single modality
  ablation: "none" # Ablation study, "none", "text_only" or "image_only"
```

3. Evaluate model:

```bash
python evaluate.py
```

This script evaluates the model with given configuration on the test set and saves the predictions. You must run the
training script first to generate the model checkpoint.

4. Ensemble results:

```bash
python ensemble.py
```

This script ensembles the results of multiple models and saves the final predictions. You must run the evaluation script
first to generate the predictions.

For the 2nd and 3rd step, you can run

```bash
./experiment.bat
```

on Windows or

```bash
chmod +x experiment.sh
./experiment.sh
```

on Linux to run the pipeline for all configurations.

## Attribution

This project cannot be completed without the following resources:

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Docs](https://scikit-learn.org/stable/index.html)
