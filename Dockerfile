# syntax=docker/dockerfile:1
FROM continuumio/miniconda3

# install dependencies
RUN apt-get update
RUN apt-get install -y wget tesseract-ocr unzip

WORKDIR /app
COPY . .

RUN python3 -m pip install --upgrade pip wheel
RUN pip3 install -r requirements.txt

# download yolov5
RUN git clone https://github.com/ultralytics/yolov5
COPY pill.yaml yolov5/data/pill.yaml

# download tesseract best model english 
RUN mkdir tessdata
RUN wget -O ./tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata?raw=true

#download pretrained model
RUN mkdir -p /app/models
RUN gdown https://drive.google.com/file/d/1-slos4_7v9bMOYFEs40HKJFJ4GI8BfzJ/view?usp=sharing --fuzzy -O /app/models/yolo.pt
RUN gdown https://drive.google.com/file/d/1-U253UBmypqAZDRJZ2fgC3hQE0-ZBSAJ/view?usp=sharing --fuzzy -O /app/models/resnet50.h5
