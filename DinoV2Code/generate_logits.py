from train import Config
from model import FungiDataset, DinoV2Lit, version_2_make_transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
import tqdm
import csv


def save_logits(model_name, output_csv_path):
    config = Config()

    model = DinoV2Lit.load_from_checkpoint(
        './checkpoints/vit-v4.ckpt',
        class_weights_dir=config.weights_dir,
        num_classes=config.num_classes,
        model_name=config.vit_model_name,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_backbone=True,
        drop_rate=0.1,
    )

    train_transforms, val_transforms = version_2_make_transforms(config.image_size)
    df = pd.read_csv(config.metadata_dir)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, config.image_path, transform=val_transforms, file_name=True, full_df=df)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["filename"] + [f"logit_{i}" for i in range(config.num_classes)]
        writer.writerow([model_name])
        writer.writerow(header)

        with torch.no_grad():
            for images, labels, filenames in tqdm.tqdm(test_loader, desc="Extracting logits"):
                images = images.to(device)
                logits = model(images)
                logits_np = logits.cpu().numpy()

                for fname, logit_row in zip(filenames, logits_np):
                    writer.writerow([fname] + logit_row.tolist())

    print(f"Logits saved to {output_csv_path}")


if __name__ == "__main__":
    ckpt='vit-v1'
    save_logits(model_name=ckpt, output_csv_path='vit_v1_test_logits.csv')
