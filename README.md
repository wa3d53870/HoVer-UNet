This repository hosts the original PyTorch implementation of HoVer-UNet, a method designed to transfer the knowledge from the multi-branch HoVerNet framework for nuclei instance segmentation and classification in histopathology images.

Features
Training
Tile Inference
WSI Inference
Dataset
PanNuke Fold1, Fold2, Fold3
CoNSeP Download
Environment Setup
To run this code, follow these steps to set up your Python environment:

Repository Structure
The repository contains the following key directories:

data/: Data loader and augmentation utilities
pannuke_metrics/: Scripts for metric calculations from Link
misc/: Miscellaneous utilities
models/: Model definitions
losses/: Loss functions used during training
train/: Training scripts
inference/: Inference functions and classes
docs/: Figures/GIFs used in the repo
Main Executable Scripts:
process_pannuke_dataset.py: Prepares the PanNuke dataset for knowledge distillation
run_train.py: Main training script
run_infer.py: Main inference script for tile inference
run_train.sh: Bash script to run training
Running the Code
Training
Data Format To train HoVer-UNet on the PanNuke dataset, run process_pannuke_dataset.py to transform the dataset into an HDF5 file for each image containing:

Original Image in RGB
Ground Truth: Nuclei map (one channel), horizontal and vertical maps (two channels), and type map (N channels, where N is the number of classes including background)
HoVerNet Predictions: Nuclei map (two channels), horizontal and vertical maps (two channels), and type map (N channels)
Execute the following command:

python process_pannuke_dataset.py --pannuke_path PANNUKE_PATH --save_path SAVE_PATH --pannuke_weights_path WEIGHTS_PANNUKE_PATH
Options:

--pannuke_path: Path to the PanNuke dataset in the provided folder structure
--save_path: Path to save the processed dataset
--pannuke_weights_path: Path to the PanNuke weights
Run Training To train HoVer-UNet, run run_train.py with the following options:


python run_train.py --base_project_dir BASE_PROJECT_DIR --project_name PROJECT_NAME --experiment_group EXPERIMENT_GROUP --experiment_id EXPERIMENT_ID --path_train PATH_TRAIN --path_val PATH_VAL --path_test PATH_TEST --pannuke_path PANNUKE_PATH [--batch_size {32,64,128,256,4,8,16}] [--nr_epochs NR_EPOCHS] [--lr LR] [--encoder ENCODER] [--use_true_labels {0,1}] [--use_hovernet_predictions {0,1}] [--loss_t {1,3,5,10,15,30}] [--loss_alpha LOSS_ALPHA]
Options:

--base_project_dir: Base directory for saving experiment results
--path_train, --path_val, --path_test: Paths to the processed PanNuke dataset for training, validation, and testing
--batch_size: Batch size for training
--nr_epochs: Number of epochs
--lr: Learning rate
--encoder: Encoder for the U-Net backbone
--use_true_labels: Whether to use ground truth labels (1 for distillation)
--use_hovernet_predictions: Whether to use HoVerNet predictions
--loss_t: Temperature coefficient for KL-divergence between student and HoVerNet predictions
--loss_alpha: Alpha for combining student and distillation loss
You can also modify and use the run_train.sh script for the training process, which runs three-fold cross-validation using different folds.

Inference
For tile inference, use run_infer.py:

python infer.py --images_path IMAGES_PATH --weights_path WEIGHTS_PATH --save_path SAVE_PATH [--step STEP] [--ext EXT] [--overlay OVERLAY]
Options:

--images_path: Path to input images
--weights_path: Path to HoVer-UNet weights
--save_path: Path to save predictions
--step: Step size for generating overlapping patches (default: 192)
--ext: Image extension (default: png)
--overlay: Whether to overlay predictions on the images (default: True)
Model Weights
Pre-trained HoVer-UNet weights can be downloaded here: https://zenodo.org/records/10101807?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU2M2JkZWYyLTgyNzgtNGM4OC05YjhkLWQwYjk1NGMyZGIxZiIsImRhdGEiOnt9LCJyYW5kb20iOiIwY2I1ZDAyZWEwODNmNTNmZGZmODM1Y2M4YTcyNGRmNSJ9.HiosnYbIK79xB-l1-CIiTi7I6yoEUd_ZVNLCmYU5qevjB7LfkZCDexqclhBQrDN1cekzNajIAa2kqjpt9kchIQ.
