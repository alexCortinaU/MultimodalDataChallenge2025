#!/bin/bash
#SBATCH --job-name=fungi_silky
#SBATCH --output=logs/train_fungi-%j.out
#SBATCH -p gpu --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

repo_dir="/home/vkr786/datasets/dikuSS25/MultimodalDataChallenge2025"
cd $repo_dir
metadata_file=$1
session_name=$2
dataset_path='data/FungiImages'
metadata_path="data/${metadata_file}"

python src/fungi_network.py $dataset_path $metadata_path --session_name $session_name