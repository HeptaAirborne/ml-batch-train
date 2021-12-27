import os
import cv2
import subprocess
import numpy as np
import yaml

from tqdm import tqdm
from pathlib import Path


def train(model_weights, model_name, classes, img_size=1280, batch_size=8, epochs=30):
    os.chdir('./yolov5')
    
    data_path = "./data/data.yaml"
    dct = {"train": "../dataset/images/train", "val": "../dataset/images/val", "nc": len(classes), "names":classes.tolist()}

    with open(data_path, "w") as f:
        yaml.dump(dct, f)
    
    # os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # subprocess.call(['python3', '-m', 'torch.distributed.launch', '--nproc_per_node=4', 'train.py', 
    #      '--weights', f"{model_weights}.pt", 
    #      '--img', str(img_size),
    #      '--batch', str(batch_size),
    #      '--epochs', str(epochs),
    #      '--data', 'data.yaml', 
    #      '--name', model_name,
    #      '--device', '0,1,2,3'])

    subprocess.call(['python3', 'train.py', 
         '--weights', f"{model_weights}.pt", 
         '--img', str(img_size),
         '--batch', str(batch_size),
         '--epochs', str(epochs),
         '--data', 'data.yaml', 
         '--name', model_name,
         '--device', '0'])


def create_dataset(annotations, dataset_type, classes):
    images_path = Path(f"./dataset/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path = Path(f"./dataset/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exist_ok=True)

    for i, img_id in enumerate(tqdm(np.unique(annotations['image_id']))):
        img = cv2.cvtColor(cv2.imread('./images/' + str(img_id) + '.jpg'), cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        os.rename('./images/' + str(img_id) + '.jpg', (images_path / (str(img_id) + '.jpg')))
        label_name = f"{img_id}.txt"

        with (labels_path / label_name).open(mode="w") as label_file:
            for index, row in annotations[annotations.image_id == img_id].iterrows():
                
                category_idx = np.where(classes == row['defect_name'])[0][0]
                
                x1, y1 = row['x_min'] / w, row['y_min'] / h
                x2, y2 = row['x_max'] / w, row['y_max'] / h
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                label_file.write(
                f"{category_idx} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
                )
    