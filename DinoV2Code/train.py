import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from model import FungiDataset, DinoV2Lit, version_2_make_transforms
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import argparse

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--sampler", choices=['random','weighted'], default='random')

        args = parser.parse_args()

        self.metadata_dir = "/home/malte/projects/MultimodalDataChallenge2025/metadata.csv"
        self.image_path = "/home/malte/datasets/FungiImages"
        self.weights_dir = "/home/malte/projects/MultimodalDataChallenge2025/class_weights.csv"
        self.vit_model_name = "vit_large_patch14_dinov2.lvd142m"
        self.epochs = 15
        self.batch_size = 32
        self.num_classes = 183
        self.learning_rate = 3e-4
        self.weight_decay = 1e-5
        self.image_size = 518
        self.num_workers = 14
        self.seed = args.seed
        self.sampler = args.sampler

def get_dataloaders(config):
    # Load metadata
    df = pd.read_csv(config.metadata_dir)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=config.seed)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    class_weights = pd.read_csv(config.weights_dir)
    class_weights = class_weights.sort_values(by="class", ascending=True)
    sample_weights = train_df['taxonID_index'].astype(int).map(
        dict(zip(class_weights['class'], class_weights['weight'] / 100))
    ).values
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True
    )
    # Initialize DataLoaders
    train_transforms, val_transforms = version_2_make_transforms(config.image_size)

    train_dataset = FungiDataset(train_df, config.image_path, transform=train_transforms)
    valid_dataset = FungiDataset(val_df, config.image_path, transform=val_transforms)
    if config.sampler == 'random':
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True
        )
    elif config.sampler == 'weighted':
            train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=sampler
        )
    else:
        raise NotImplementedError('WRONG SAMPLER')
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, valid_loader

def pl_trainer(config):
    train_loader, valid_loader = get_dataloaders(config)
    model = DinoV2Lit(
        class_weights_dir=config.weights_dir,
        num_classes=config.num_classes,
        model_name=config.vit_model_name,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        drop_rate=0.1,
    )

    tb_logger = TensorBoardLogger(
        save_dir="logs",          
        name="vit",            
    )

    # --- Checkpointing (save BEST and LAST) ---
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",  # save checkpoints here
        monitor="val_loss",       
        mode="min",             
        save_top_k=1,         
        save_last=True,           # also keep last checkpoint
        filename="vit",  # customize as you like
    )

    # (optional) LR monitor & early stopping
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",           # e.g., 1 or [0] or "auto"
        precision="16-mixed",     # or "32-true"
        logger=tb_logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        log_every_n_steps=1,
    )

    # --- Train ---
    trainer.fit(model, train_loader, valid_loader)
    print("Best ckpt:", checkpoint_cb.best_model_path)

if __name__ == "__main__":
    config = Config()
    pl_trainer(config)