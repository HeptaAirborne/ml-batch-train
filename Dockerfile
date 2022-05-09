FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /hepta
ADD . /hepta
RUN pip install -r requirements.txt
RUN pip install -r yolov5/requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p inference/results/labels/
ENV AWS_DEFAULT_REGION=eu-west-1
