import pandas as pd
import json
import boto3

from multiprocessing.dummy import Pool as ThreadPool
from boto3.session import Session
from botocore.exceptions import ClientError
from io import BytesIO


def get_ssm_secret(parameter_name, aws_key, aws_secret):
    ssm = boto3.client("ssm", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    parameter = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
    return parameter.get("Parameter").get("Value")


def download(image_url, image_id, aws_key, aws_secret):
    bucket, key = image_url
    session = Session(aws_access_key_id=aws_key,
                      aws_secret_access_key=aws_secret)
    s3 = session.resource('s3')
    print('Downloading image %s from bucket %s' % (key, bucket))
    try: 
        request = s3.Object(bucket, key)
        contents = request.get()['Body'].read()
        img_name = f"{str(image_id)}.jpg"
        with open('./images/' + img_name, 'wb') as f:
            f.write(contents)
        return contents
    except ClientError as e:
        print(f"Error occured while downloading {image_url}: {e}")


def download_images(image_list, image_ids, aws_key, aws_secret):
    pool = ThreadPool(4)
    arglist = list(zip(image_list, image_ids,
                       [aws_key] * len(image_list),
                       [aws_secret] * len(image_list)))
    results = pool.starmap(download, arglist)
    pool.close()
    pool.join()
    return results


def split_bucket_and_key(url):
    path = url.split('//')[1]
    components = path.split('/')
    bucket = components[0]
    key = '/'.join(components[1:])
    return bucket, key


def get_input_df(aws_access_key_id, aws_secret_access_key, csv_bucket, csv_key):
    session = Session(aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)

    s3 = session.resource('s3', use_ssl=False)
    obj = s3.Object(csv_bucket, csv_key).get()
    temp = BytesIO(obj['Body'].read())
    df = pd.read_csv(temp)
    return df


def read_json_from_s3(s3_obj, bucket, file_path):
    obj = s3_obj.Object(bucket, file_path)
    data = obj.get()['Body'].read().decode('utf-8')
    json_data = json.loads(data) 
    return json_data


def upload_file_to_s3(s3_obj, bucket_name, s3_path, file_path, file_name):
    bucket = s3_obj.Bucket(bucket_name)
    bucket.upload_file(file_path, s3_path + file_name)