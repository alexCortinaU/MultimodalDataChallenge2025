from train import Config
from model import FungiDataset, DinoV2Lit, version_2_make_transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
import tqdm
import csv

def test_model(session_name, output_csv_path):
    config = Config()

    # Initialize the model
    model = DinoV2Lit.load_from_checkpoint('./checkpoints/vit-v4.ckpt',
                                            class_weights_dir=config.weights_dir,
                                            num_classes=config.num_classes,
                                            model_name=config.vit_model_name,
                                            lr=config.learning_rate,
                                            weight_decay=config.weight_decay,
                                            freeze_backbone=True,
                                            drop_rate=0.1,)

    train_transforms, val_transforms = version_2_make_transforms(config.image_size)
    df = pd.read_csv(config.metadata_dir)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, config.image_path, transform=val_transforms, file_name=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    # test_model(session_name='vit_version_1', output_csv_path='vit_version_1.csv') # pretrained vit on df
    test_model(session_name='baseline_vit_v4', output_csv_path='vit_v4.csv') # pretrained vit on df