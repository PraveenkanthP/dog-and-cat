import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(api, dataset_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    api.dataset_download_files(dataset_name, path=save_path, unzip=True)

def organize_dataset(dataset_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    
    # Move cat images
    os.makedirs(os.path.join(target_path, 'training_set', 'cats'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'test_set', 'cats'), exist_ok=True)
    for i in range(1, 12501):
        os.rename(os.path.join(dataset_path, f'train/cat.{i}.jpg'), 
                  os.path.join(target_path, 'training_set', 'cats', f'cat{i}.jpg'))
    for i in range(12501, 12501+2500):
        os.rename(os.path.join(dataset_path, f'train/cat.{i}.jpg'), 
                  os.path.join(target_path, 'test_set', 'cats', f'cat{i}.jpg'))
    
    # Move dog images
    os.makedirs(os.path.join(target_path, 'training_set', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'test_set', 'dogs'), exist_ok=True)
    for i in range(1, 12501):
        os.rename(os.path.join(dataset_path, f'train/dog.{i}.jpg'), 
                  os.path.join(target_path, 'training_set', 'dogs', f'dog{i}.jpg'))
    for i in range(12501, 12501+2500):
        os.rename(os.path.join(dataset_path, f'train/dog.{i}.jpg'), 
                  os.path.join(target_path, 'test_set', 'dogs', f'dog{i}.jpg'))

def main():
    # Specify dataset name and save path
    dataset_name = 'prasunroy/natural-images'
    save_path = 'dataset'

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    download_dataset(api, dataset_name, save_path)

    # Organize dataset
    dataset_path = os.path.join(save_path, 'train')
    target_path = os.path.join(save_path)
    organize_dataset(dataset_path, target_path)

if __name__ == "__main__":
    main()
