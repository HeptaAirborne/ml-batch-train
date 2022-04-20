import numpy as np
import argparse
import os
import json

from boto3.session import Session
from sklearn.model_selection import train_test_split

from utils.yolov5_utils import train, create_dataset
from utils.aws_utils import download_images, split_bucket_and_key, get_input_df, read_json_from_s3, upload_file_to_s3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws_id", type=str, default="", help="aws id")
    parser.add_argument("--aws_secret", type=str, default="", help="aws secret")

    opt = parser.parse_args()

    session = Session(aws_access_key_id=opt.aws_id,
                      aws_secret_access_key=opt.aws_secret)
    s3 = session.resource('s3', verify=False, use_ssl=False)

    data = read_json_from_s3(s3, 'hepta-ml-model-weights', 'config/train_params.json')
    annotations = get_input_df(opt.aws_id, opt.aws_secret, data['csv_bucket'], data['csv_key'])

    models_config = read_json_from_s3(s3, 'hepta-ml-model-weights', 'config/models_config.json')

    if data['model_name'] in models_config:
        raise RuntimeError(f"The model with name {data['modelKey']} already exist in models_config.json")

    CLASSES = np.array([cl['name'] for cl in data['classes']])
    MODEL_NAME = data['model_name']
    MODEL_TYPE = data['model_type']
    IMG_SIZE = data['img_size']
    EPOCHS = data['epochs']
    BACTH_SIZE = data['batch_size']
    TEST_SET_SIZE = data['test_set_size']

    print('Downloading images...')
    os.mkdir('./images')
    unique_df = annotations.drop_duplicates('image_id')
    image_paths = unique_df.image_path.values
    image_list = [split_bucket_and_key(x) for x in image_paths]
    image_ids = unique_df.image_id.values
    imglist_bytes = download_images(image_list, image_ids, opt.aws_id, opt.aws_secret)
    print('Images downloaded.')

    print("Creating datasets...")
    train_data, val_data = train_test_split(np.unique(annotations['image_id']), test_size=TEST_SET_SIZE)
    print(f'Train set size: {len(train_data)}, test set size: {len(val_data)}')

    create_dataset(annotations[annotations.image_id.isin(train_data)], 'train', CLASSES)
    create_dataset(annotations[annotations.image_id.isin(val_data)], 'val', CLASSES)
    print('Datasets created.')
    
    print( val_data['image_id'] )

    # print("Training model...")
    # try:
    #     train(MODEL_TYPE, MODEL_NAME, CLASSES, IMG_SIZE, BACTH_SIZE, EPOCHS)
    # except Exception as e:
    #     print("Error occured: {e}")
    # print("Training finished.")

    # print("Saving model weights to S3...")
    # upload_file_to_s3(s3, 'hepta-ml-model-weights', '', f'./runs/train/{MODEL_NAME}/weights/best.pt',
    #                   MODEL_NAME + '.pt')

    # print("Adding model setting to models_config.json")
    # models_config[MODEL_NAME] = {'classes': data['classes']}

    # with open('models_config.json', 'w') as f:
    #     json.dump(models_config, f)

    # upload_file_to_s3(s3, 'hepta-ml-model-weights', 'config/', './models_config.json', 'models_config.json')
    print("Job completed!")
