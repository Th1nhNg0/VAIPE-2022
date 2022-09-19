import pandas as pd
import pytesseract
import re
import PIL
import difflib
import numpy as np
import random
import json
from glob import glob
from tqdm import tqdm
from PIL import ImageFile
import torch
import tensorflow as tf
import argparse
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def is_similar(first, second, ratio):
    return difflib.SequenceMatcher(None, first, second).ratio() > ratio
    
def findid(text,pres_df):
    df = pd.DataFrame([text],columns =['text'])
    result = [s for f in df['text'] for s in pres_df['text'] if is_similar(f,s, 0.8)]
    df = pd.DataFrame(result,columns =['text'])['text'].unique()
    df = pd.DataFrame(df,columns =['text'])
    if not df.empty:
      # print(df.merge(pres_df,on = 'text' )['mapping'].values)
      return df.merge(pres_df,on = 'text' )['mapping'].unique().tolist() # Chỉnh khúc này nếu muốn dùng xác xuất
    else:
      return None

def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def convert(txt:str):
    txt = txt.lower()
    txt = remove_accents(txt)
    txt = re.sub(r'^\d+\s*\)\s*', '', txt)
    txt = re.sub(r'\s*sl:.+', '', txt)
    return txt

def pres_process(pres_path:str):
    config_tesseract = "--tessdata-dir tessdata"

    txt=pytesseract.image_to_string(pres_path,lang='eng',config=config_tesseract)
    match = re.findall(r'^\d\s*\)\s*\w.+', txt,flags=re.MULTILINE|re.U)
    # remove 1) and SL: VIEN
    match = list(map(lambda x: convert(x), match))
    # convert to id
    ids = list(map(lambda x: findid(x,pres_df), match))
    ids = [item for sublist in ids for item in sublist]
    return match, ids

def classification(img_path:str,ids)->pd.DataFrame:
    result = yolo_model(img_path,size=1280).pandas().xyxy[0]
    img = PIL.Image.open(img_path)
    img = PIL.ImageOps.exif_transpose(img)
    # img.show()
    for index,row in result.iterrows():
        # crop img
        x1,y1,x2,y2 = row['xmin'],row['ymin'],row['xmax'],row['ymax']
        crop = img.crop((x1,y1,x2,y2))
        # PIL image to tensorflow image
        crop = tf.convert_to_tensor(np.array(crop))
        crop = tf.image.resize(crop,(IMAGE_WIDTH,IMAGE_HEIGHT))
        crop = tf.image.convert_image_dtype(crop,tf.float32)
        crop = crop.numpy()
        crop = crop.reshape((1,)+crop.shape)
        predict = class_model.predict([ids,crop],verbose=0)[0].argmax()
        result.loc[index,'class_id'] = predict
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--yolo_model', type=str, help='path to yolov5 model',nargs='?', const='/app/models/yolo.pt',default='/app/models/yolo.pt')
    parser.add_argument('--class_model', type=str, help='path to classification model',nargs='?', const='/app/models/resnet50.h5',default='/app/models/resnet50.h5')
    
    args = parser.parse_args()
    yolo_model_path = args.yolo_model
    class_model_path = args.class_model


    pres_df = pd.read_csv('/app/pres_df.csv')
    yolo_model = torch.hub.load('ultralytics/yolov5','custom',path=yolo_model_path)
    print("✅ LOADING YOLO MODEL DONE")
    class_model = tf.keras.models.load_model(class_model_path)
    print("✅ LOADING CLASSIFICATION MODEL DONE")

 
    f = open('/app/public_test/pill_pres_map.json', 'r')
    data = json.load(f)
    result = []
    for key in data:
        pres = key+'.png'
        pill = data[key]
        result.append([pres,pill])

    df = pd.DataFrame(result,columns =['pres','pill'])
    df['pres']=df['pres'].apply(lambda x:  '/app/public_test/prescription/image/'+x)
    df['pill']=df['pill'].apply(lambda arr: list(map(lambda x: '/app/public_test/pill/image/'+x,arr)))

    print('{} prescription images\n{} pill images'.format(len(df['pres'].unique()),df['pill'].apply(lambda x: len(x)).sum()))
    pbar = tqdm(total=df['pill'].apply(lambda x: len(x)).sum())

    
    predict = pd.DataFrame()
    for index,row in df.iterrows():
        pres_path = row['pres']
        pill_paths = row['pill']
        pill_names,ids=pres_process(pres_path)
        pill_ids = [0]*108
        pill_ids[107] = 1
        for id in ids:
            pill_ids[id] = 1
        pill_ids = np.array(pill_ids)
        pill_ids = pill_ids.reshape((1,)+pill_ids.shape)
        for pill_path in pill_paths:
            temp = classification(pill_path,pill_ids)
            temp['image_name'] = pill_path.split('/')[-1]
            predict = pd.concat([predict,temp])
            pbar.update(1)

    predict.drop(['class','name'], axis=1, inplace=True)
    predict.rename(columns={'confidence':'confidence_score',
                        'xmin':'x_min',
                        'ymin':'y_min',
                        'xmax':'x_max',
                        'ymax':'y_max'
                        },inplace=True)
    predict = predict[['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max']]
    predict[['class_id','x_min','y_min','x_max','y_max']] = predict[['class_id','x_min','y_min','x_max','y_max']].astype(int)
    predict['confidence_score'] = predict['confidence_score'].round(2)
    predict.to_csv('/app/results.csv',index=False)
    print("✅ DONE,save file to /app/results.csv")