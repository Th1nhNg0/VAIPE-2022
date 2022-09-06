# syntax=docker/dockerfile:1
FROM ubuntu:20.04

# install dependencies
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y python3-pip git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3.8-dev
RUN apt install -y wget
RUN apt install -y tesseract-ocr


WORKDIR /app
COPY . .

# install python package
RUN python3 -m pip install --upgrade pip wheel
RUN pip3 install -r requirements.txt

# download yolov5
RUN git clone https://github.com/ultralytics/yolov5
COPY pill.yaml yolov5/data/pill.yaml

#download tesseract best model english 
RUN mkdir tessdata
RUN wget -O ./tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata?raw=true


# CMD [ "python3", "code/inference.py"]
