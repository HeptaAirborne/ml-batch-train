import pandas as pd
import numpy as np

IMAGES_PATH = './images/'
# S3_ANNOTATIONS_PATH = 'input/annotations/'

YOLO_MODEL_PATH = 'weights/model.pt'
PATH_TO_PRED_LABELS = './inference/results/labels/'
ELEMENT_TYPE_ID = None
FAILED_IMAGES_URLS = []

IMG_SIZE = 1280
CONF_THRESHOLD = 0.4

MODELS_OUTPUT_TO_UBIRD_FORM = {
					'rust_detector': {
						'classes': [
							{'name': 'rust', 'annotation_type_id': 113, 'severity': 3},
							{'name': 'normal', 'annotation_type_id': 113, 'severity': 0}
						]
					},
					'corrosion_rotten_detector': {
						'classes': [
							{'name': 'corrosion', 'annotation_type_id': 470, 'severity': 3},
							{'name': 'rotten', 'annotation_type_id': 471, 'severity': 3},
							{'name': 'normal', 'annotation_type_id': 471, 'severity': 0},
						]
					},
					'ca_crooked_detector': {
						'classes': [
							{'name': 'crooked', 'annotation_type_id': 21, 'severity': 3},
							{'name': 'rotten', 'annotation_type_id': 19, 'severity': 0}
						]
					},
					'insulator_detector': {
						'classes': [
							{'name': 'insulator', 'annotation_type_id': 9, 'severity': 0}
						]
					}
				}


def output_to_ubird_format(df, model_key):

	model_config = MODELS_OUTPUT_TO_UBIRD_FORM[model_key]

	for i, c in enumerate(model_config['classes']):
		df.loc[df['class'] == i, 'annotation_type_id'] = c['annotation_type_id']
		df.loc[df['class'] == i, 'severity'] = c['severity']

	df['severity'] = df['severity'].astype(np.int32)
	df = df[['image_url', 'artifact_id', 
             'x_min', 'y_min', 'x_max', 'y_max', 
             'annotation_type_id', 'confidence', 'severity']]

	return df
	
