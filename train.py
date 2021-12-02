import numpy as np
import pandas as pd
import argparse
import os
import cv2
import random
import string
import requests

from tqdm import tqdm
from boto3.session import Session

import utils.config as config
from utils.yolov5_utils import train, create_dataset
from utils.aws_utils import split_bucket_and_key, download_images, download_model, upload_file_to_s3, get_ssm_secret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="", help="uBird API annotations url")
    parser.add_argument("--environment", type=str, default="", help="uBird environment")
    parser.add_argument("--aws_id", type=str, default="", help="aws id")
    parser.add_argument("--aws_secret", type=str, default="", help="aws secret")
    
    opt = parser.parse_args()
    
    session = Session(aws_access_key_id=opt.aws_id,
                      aws_secret_access_key=opt.aws_secret)
    s3 = session.resource('s3', verify=False, use_ssl=False)
    
    param_name = f"/ubird/{opt.environment}/ML_API_KEY"
    ml_api_key = get_ssm_secret(param_name, opt.aws_id, opt.aws_secret)
    
    resp = requests.get(url=opt.train_url, headers={'Authorization': f'Bearer {ml_api_key}'})
    data = resp.json()
    
    # status = 'RUNNING'
    # requests.put(url=opt.train_url + '/status', json=status, headers={'Authorization': f'Bearer {ml_api_key}', 'Content-Type': 'application/json'})
    
    annotations = pd.DataFrame(data['annotations'])
    annotations.defect_name = annotations.defect_name.str.lower()

    CLASSES = np.sort(annotations.defect_name.unique())
    MODEL_NAME = data['model_name']
    MODEL_TYPE = data['model_type']
    IMG_SIZE = data['epochs']
    EPOCHS = data['epochs']
    BACTH_SIZE = data['batch_size']

    print('Downloading images...')
    os.mkdir('./images')
    image_names = unique_df.image_path.values
    image_list = [split_bucket_and_key(x) for x in image_names]
    image_ids = unique_df.image_id.values
    imglist_bytes = download_images(image_list, image_ids, aws_id, aws_secret)
    print('Images downloaded.')
    
    print("Creating datasets...")
    train_data, val_data = train_test_split(np.unique(annotations['image_id']), test_size=0.15)
    print(f'Train set size: {len(train_data)}, test set size: {len(val_data)}')

    create_dataset(annotations[annotations.image_id.isin(train_data)], 'train')
    create_dataset(annotations[annotations.image_id.isin(val_data)], 'val')
    print('Datasets created.')

    print("Training model...")
    train(MODEL_TYPE, MODEL_NAME, IMG_SIZE, BACTH_SIZE, EPOCHS)
    print("Training finished.")

    print("Saving model weights to S3...")
    upload_file_to_s3(s3, 'hepta-ml-model-weights', '', f'./yolov5/runs/train/{MODEL_NAME}/weights/best.pt', MODEL_NAME)
    print("Job completed!")