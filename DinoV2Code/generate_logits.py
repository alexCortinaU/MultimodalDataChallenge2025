from train import Config
from model import FungiDataset, DinoV2Lit, version_2_make_transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
import tqdm
import csv


def save_logits_and_embeddings(model_name, logits_csv_path, emb_csv_path):
    config = Config()

    model = DinoV2Lit.load_from_checkpoint(
        f'./checkpoints/{model_name}.ckpt',
        class_weights_dir=config.weights_dir,
        num_classes=config.num_classes,
        model_name=config.vit_model_name,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_backbone=True,
        drop_rate=0.1,
    )

    _, val_transforms = version_2_make_transforms(config.image_size)
    df = pd.read_csv(config.metadata_dir)
    dataset = FungiDataset(df, config.image_path, transform=val_transforms, file_name=True, full_df=df)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=12)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Get embedding dim for header
    with torch.no_grad():
        dummy_out = model.backbone(torch.zeros(1, 3, config.image_size, config.image_size).to(device))
        emb_dim = dummy_out.shape[1]

    # Prepare CSV writers
    logits_file = open(logits_csv_path, mode="w", newline="")
    emb_file = open(emb_csv_path, mode="w", newline="")

    logits_writer = csv.writer(logits_file)
    emb_writer = csv.writer(emb_file)

    logits_writer.writerow([model_name])
    emb_writer.writerow([model_name])

    logits_writer.writerow(["filename"] + [f"logit_{i}" for i in range(config.num_classes)])
    emb_writer.writerow(["filename"] + [f"emb_{i}" for i in range(emb_dim)])

    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(dataloader, desc="Extracting logits & embeddings"):
            images = images.to(device)
            feats = model.backbone(images)  # embeddings
            logits = model.head(feats)      # logits

            feats_np = feats.cpu().numpy()
            logits_np = logits.cpu().numpy()

            for fname, logit_row, emb_row in zip(filenames, logits_np, feats_np):
                logits_writer.writerow([fname] + logit_row.tolist())
                emb_writer.writerow([fname] + emb_row.tolist())

    logits_file.close()
    emb_file.close()
    print(f"Logits saved to {logits_csv_path}")
    print(f"Embeddings saved to {emb_csv_path}")


if __name__ == "__main__":
    ckpt = 'vit_f1.ckpt'
    save_logits_and_embeddings(
        model_name=ckpt,
        logits_csv_path=f'{ckpt}_logits.csv',
        emb_csv_path=f'{ckpt}_embeddings.csv'
    )
