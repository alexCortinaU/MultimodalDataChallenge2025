#!/bin/bash
#SBATCH --job-name=fungi_silky
#SBATCH --output=logs/train_fungi-%j.out
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

repo_dir="/home/vkr786/datasets/dikuSS25/MultimodalDataChallenge2025"
cd $repo_dir
metadata_file=$1
session_name=$2
dataset_path='data/FungiImages'
metadata_path="data/${metadata_file}"
workers=32
batch_size=32
model_name='efficientnet'
output_file_name='val_predictions.csv'

python src/eval_network.py $dataset_path $metadata_path --session_name $session_name \
--workers $workers --batch_size $batch_size --model $model_name --outputfile $output_file_name \
--is_validation