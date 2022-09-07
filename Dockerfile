# syntax=docker/dockerfile:1
FROM continuumio/miniconda3

# install dependencies
RUN apt-get update
RUN apt-get install -y wget tesseract-ocr unzip

WORKDIR /app
COPY . .

# download yolov5
RUN git clone https://github.com/ultralytics/yolov5
COPY pill.yaml yolov5/data/pill.yaml

# download tesseract best model english 
RUN mkdir tessdata
RUN wget -O ./tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata?raw=true
