import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import (RandomResizedCrop, 
                            HorizontalFlip, 
                            VerticalFlip, 
                            RandomBrightnessContrast,
                            GaussianBlur)
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter
import argparse
from pathlib import Path
from monai.losses import FocalLoss
this_path = Path().resolve()

def parse_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('dataset', type=str, help='Image folder with dataset to process')
    parser.add_argument('metadata', type=str, help='Path to the metadata file')
    parser.add_argument('--session_name', '-s', type=str, default='EfficientNet', help='Session name for the experiment')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for train/test.')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'vit_b'], help='Model to use for training')
    parser.add_argument('--is_validation', action='store_true', help='Flag to indicate if the evaluation is for validation set')
    parser.add_argument('--outputfile', type=str, default='test_predictions.csv', help='Name of the output file for predictions')

    return parser.parse_args()

def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop(size=(width, height), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            GaussianBlur(sigma_limit=(0.5, 2.0),
                         blur_limit=(6, 6),
                         p=0.5
                         ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df['taxonID_index'].values[idx]  # Get label
        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            # Convert to RGB mode (handles grayscale images as well)
            image = img.convert('RGB')
        image = np.array(image)

        # Apply transformations if available
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, file_path

def evaluate_network_on_test_set(data_file,
                                 image_path,
                                 checkpoint_dir,
                                 session_name,
                                 workers=8,
                                 batch_size=32,
                                 model_name='efficientnet',
                                 is_validation=False,
                                 output_file_name='test_predictions.csv'):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    
    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, output_file_name)

    df = pd.read_csv(data_file)
    if is_validation:
        test_df = df[df['split'] == 'val']
    else:
        test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'efficientnet':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, 183)
        )
    elif model_name == 'vit_b':
        model = models.vit_b_16(weights='DEFAULT')
        model.heads = nn.Sequential(
            nn.Linear(model.hidden_dim, 183)
        )
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # Path to fungi images
    image_path = args.dataset
    # Path to metadata file
    data_file = args.metadata
    assert Path(data_file).exists(), f"Metadata file not found: {data_file}"
    assert Path(image_path).exists(), f"Image folder not found: {image_path}"

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = args.session_name

    # Folder for results of this experiment based on session name:
    checkpoint_dir = this_path / "checkpoints" / session
    print(f'Checkpoint directory to run evaluation from: {checkpoint_dir}')

    evaluate_network_on_test_set(str(data_file),
                                 str(image_path),
                                 str(checkpoint_dir),
                                 session,
                                 workers=args.workers,
                                 batch_size=args.batch_size,
                                 is_validation=args.is_validation,
                                 model_name=args.model,
                                 output_file_name=args.outputfile)