import random
from glob import glob

import cv2
import paddlehub as hub
import pandas as pd
import os
from tqdm import tqdm



def infer(image):
    result = model.Segmentation(
        images=[image],
        batch_size=1,
        )
    return result[0]['front'][:,:,::-1], result[0]['mask']

def checkBox(x,y,w,h,orgW,orgH):
    ratio = 0.25
    x1=x
    x2=x+w
    y1=y
    y2=y+h
    left,right,up,down=True,True,True,True
    if x1/orgW>ratio:
        left=False
    if (orgW-x2)/orgW>ratio:
        right=False
    if y1/orgH>ratio:
        up=False
    if (orgH-y2)/orgH>ratio:
        down=False
    return left,right,up,down

def findBBOX(image):
    _,img2=infer(image)
    _,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = -1,-1,-1,-1
    if len(cnts) != 0:
        c = max(cnts, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
    return x,y,w,h

if __name__=='__main__':
    model = hub.Module(name='U2Net')
    image_paths = glob('/app/public_train/pill/image/*')
    save_path = '/app/gen_train_data/new_label/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        label_path = image_path.replace('image','label').replace('jpg','json')
        label_name = label_path.split('/')[-1]
        label = pd.read_json(label_path)
        for i,row in label.iterrows():
            x = row['x']
            y = row['y']
            w = row['w']
            h = row['h']
            crop_img = img[y:y+h, x:x+w]
            nx,ny,nw,nh=findBBOX(crop_img)
            if nw !=-1 and nh != -1 and all(checkBox(nx,ny,nw,nh,crop_img.shape[1],crop_img.shape[0])):
                nx+=x
                ny+=y
                label.at[i,'x']=nx
                label.at[i,'y']=ny
                label.at[i,'w']=nw
                label.at[i,'h']=nh
        label.to_json(f'/app/gen_train_data/new_label/{label_name}',orient="records")
