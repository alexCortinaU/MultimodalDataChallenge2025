import torch
import torch.nn as nn
import timm
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sentence_transformers import SentenceTransformer
from lora import QkvWithLoRA
from sklearn.decomposition import PCA
from functools import partial

class DinoV2Lit(L.LightningModule):
    """
    DINOv2 backbone (ViT) + linear head for multiclass classification.
    Defaults to vit_base_patch14_dinov2.lvd142m.
    """
    def __init__(
        self,
        class_weights_dir: str,
        num_classes: int,
        model_name: str,
        lr: float,
        weight_decay: float,
        drop_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create backbone with no classifier head; we'll add our own.
        # timm will load DINOv2 weights for these model names.

        self.backbone = self._init_backbone(model_name)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, num_classes),
        )

        class_weights = pd.read_csv(class_weights_dir)
        class_weights = class_weights.sort_values(by="class", ascending=True)
        print("class weights head: ", class_weights.head())
        self.class_weights = class_weights["weight"].values
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).float())
        
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_acc1   = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc5   = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_acc1  = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.test_acc5  = MulticlassAccuracy(num_classes=num_classes, top_k=5)

        self.train_f1 = MulticlassF1Score(num_classes=num_classes)
        self.val_f1   = MulticlassF1Score(num_classes=num_classes)
        self.test_f1  = MulticlassF1Score(num_classes=num_classes)

    @staticmethod
    def _init_backbone(model_name):
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # gets feature embeddings
        )
        assign_lora = partial(QkvWithLoRA, rank=8, alpha=1)
        for block in model.blocks:
            block.attn.qkv = assign_lora(block.attn.qkv)

        for param in model.parameters():
            param.requires_grad = False

        for block in model.blocks:
            for param in block.attn.qkv.lora_q.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_v.parameters():
                param.requires_grad = True
        return model
    
    def on_train_start(self):
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.logger.log_dir)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch == 0 and batch_idx == 0:
            imgs, labels = batch
            grid = torchvision.utils.make_grid(imgs, normalize=True, scale_each=True)
            self.logger.experiment.add_image(
                "first_epoch_first_batch", grid, global_step=0
            )

    def forward(self, x):
        # ViT forward_features returns CLS embedding [B, C] for timm ViTs
        feats = self.backbone(x)   # [B, C]
        logits = self.head(feats)                   # [B, num_classes]
        return logits

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.long().view(-1))

        if stage == "train":
            self.train_acc1.update(logits, y)
            self.train_acc5.update(logits, y)
            self.train_f1.update(logits, y)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        elif stage == "val":
            self.val_acc1.update(logits, y)
            self.val_acc5.update(logits, y)
            self.val_f1.update(logits, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.test_acc1.update(logits, y)
            self.test_acc5.update(logits, y)
            self.test_f1.update(logits, y)
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train/acc1", self.train_acc1.compute(), prog_bar=True)
        self.log("train/acc5", self.train_acc5.compute(), prog_bar=False)
        self.log("train/f1", self.train_f1.compute(), prog_bar=False)
        self.train_acc1.reset(); self.train_acc5.reset(); self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc1", self.val_acc1.compute(), prog_bar=True)
        self.log("val/acc5", self.val_acc5.compute(), prog_bar=False)
        self.log("val/f1", self.val_f1.compute(), prog_bar=False)
        self.val_acc1.reset(); self.val_acc5.reset(); self.val_f1.reset()

    def on_test_epoch_end(self):
        self.log("test/acc1", self.test_acc1.compute(), prog_bar=True)
        self.log("test/acc5", self.test_acc5.compute(), prog_bar=False)
        self.log("test/f1", self.test_f1.compute(), prog_bar=False)
        self.test_acc1.reset(); self.test_acc5.reset(); self.test_f1.reset()    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.hparams.lr * 0.1},
            {'params': self.head.parameters(), 'lr': self.hparams.lr}
        ], weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10000, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }


def version_2_make_transforms(image_size: int = 224):
    mean = (0.485, 0.456, 0.406)  # standard ImageNet
    std  = (0.229, 0.224, 0.225)
    

    train_tf = A.Compose([

        # geometry: small + plausible
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=25,
                        border_mode=1, p=0.7),   # reflection padding
        A.HorizontalFlip(p=0.5),                    # most fungi have no left/right bias
        A.VerticalFlip(p=0.5),                      # use if orientation isn’t meaningful
        A.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),

        # photometric: conservative to keep species colour cues
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=6, p=0.3),
        A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=0.3),

        # camera/real-world noise (light)
        A.ImageCompression(quality_range=(75, 100), p=0.2),
        A.GaussNoise(std_range=(0.01,0.09), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),

        # occlusion (very light; can help robustness)
        A.CoarseDropout(num_holes_range=(1, 2),
                        hole_height_range=(0.06, 0.12),
                        hole_width_range=(0.06, 0.12),
                        fill=0, p=0.2),

        # normalize to your backbone’s stats (example: ImageNet)
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return train_tf, val_tf

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None, file_name=False, full_df=None):
        self.df = df
        self.transform = transform
        self.path = path
        self.file_name = file_name

        if full_df is not None:
            self.habitat_emb_map = self._get_embedding_map(full_df['Habitat'].unique())
            self.substrate_emb_map = self._get_embedding_map(full_df['Substrate'].unique())

    def _get_embedding_map(self, tokens, n_components=12):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(tokens)
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)
        embedding_map = dict(zip(tokens, embeddings))
        return embedding_map

    def __len__(self):
        return len(self.df)

    def _meta_data_encoder(self, row):
        habitat_embedding = self.habitat_emb_map[row['Habitat']]
        substrate_embedding = self.substrate_emb_map[row['Substrate']]
        return np.concat([habitat_embedding, substrate_embedding])

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]

        embeddings = self._meta_data_encoder(self.df.iloc[idx])
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
            image = self.transform(image = image)
        if self.file_name:
            return image['image'], label, file_path
        else:
            return image['image'], label