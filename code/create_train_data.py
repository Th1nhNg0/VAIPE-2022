import difflib
from functools import partial
import os
import random
import re
from glob import glob
from multiprocessing import Pool

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def CREATE_PRESS_DF():
    print("RUNNING PRESS_DF DATA CREATION")
    files = glob('/app/public_train/prescription/label/*.json')
    pres_df = pd.DataFrame()
    for file in tqdm(files,desc='CREATING PRESS_DF DATA'):
        temp_df = pd.read_json(file)
        temp_df['file_name'] = file.split('\\')[-1]
        pres_df = pd.concat([pres_df,temp_df])
    pres_df = pres_df[pres_df['label']=='drugname']
    pres_df['mapping'] = pres_df['mapping'].astype(int)
    pres_df['text']=pres_df['text'].str.replace(r'^\d+\s*\)\s+','',regex=True).str.lower().apply(lambda x: remove_accents(x))
    pres_df.drop(['id'], axis=1, inplace=True)
    pres_df.head()
    pres_df.to_csv('/app/pres_df.csv')

def process_img(i,file,valid_len,save_path_image_val,save_path_image_train,save_path_label_val,save_path_label_train):
    img_path = file.replace('label', 'image').replace('.json', '.jpg')
    img = Image.open(img_path)
    try:
        img = ImageOps.exif_transpose(img)
    except:
        pass
    W,H = img.size
    new_width = 640
    new_height = int(new_width * H / W)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    df = pd.read_json(file)
    txt = ''
    for index, row in df.iterrows():
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        x = x /W
        y = y /H
        w = w /W
        h = h /H
        x_center = (x + w / 2)
        y_center = (y + h / 2) 
        w_ratio = w
        h_ratio = h 
        # to 6 decimal places
        txt += '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(0,x_center, y_center, w_ratio, h_ratio)
    base_name = os.path.basename(img_path)
    if i > valid_len:
        img.save(save_path_image_train + base_name)
        with open(save_path_label_train + base_name.replace('.jpg', '.txt'), 'w') as f:
            f.write(txt)
    else:
        img.save(save_path_image_val + base_name)
        with open(save_path_label_val + base_name.replace('.jpg', '.txt'), 'w') as f:
            f.write(txt)

def YOLO_DATASET():
    file_path = '/app/public_train/pill/label/*.json'
    save_dir='/app/gen_train_data/yolo_pill'
    split_ratio=0.1
    files = glob(file_path)
    # shuffle the files
    random.shuffle(files)
    save_path_image = f'{save_dir}/images'
    save_path_label = f'{save_dir}/labels'
    if not os.path.exists(save_path_image):
        os.makedirs(save_path_image)
    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)
    # val and train folder
    save_path_image_val = f'{save_dir}/images/val/'
    save_path_label_val = f'{save_dir}/labels/val/'
    if not os.path.exists(save_path_image_val):
        os.makedirs(save_path_image_val)
    if not os.path.exists(save_path_label_val):
        os.makedirs(save_path_label_val)
    save_path_image_train = f'{save_dir}/images/train/'
    save_path_label_train = f'{save_dir}/labels/train/'
    if not os.path.exists(save_path_image_train):
        os.makedirs(save_path_image_train)
    if not os.path.exists(save_path_label_train):
        os.makedirs(save_path_label_train)

    train_len = int(len(files))
    valid_len = int(train_len * split_ratio)
    print(f'train: {train_len-valid_len} images\nvalid: {valid_len} images')

    jobs=[
        (i,file,valid_len,save_path_image_val,save_path_image_train,save_path_label_val,save_path_label_train,) for i,file in enumerate(files)
    ]
    
    pool = Pool()
    print("STARTING PROCESSING WITH {} PROCESSES".format(pool._processes))

    pbar = tqdm(total=len(jobs),desc='CREATING YOLO TRAIN DATA')
    for i in range(len(jobs)):
        pool.apply_async(process_img, args=jobs[i], callback=lambda x: pbar.update())
    pool.close()
    pool.join()
    pbar.close()

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

def is_similar(first, second, ratio):
    return difflib.SequenceMatcher(None, first, second).ratio() > ratio
    
def findid(text,pres_df):
    df = pd.DataFrame([text],columns =['text'])
    result = [s for f in df['text'] for s in pres_df['text'] if is_similar(f,s, 0.9)]
    df = pd.DataFrame(result,columns =['text'])['text'].unique()
    df = pd.DataFrame(df,columns =['text'])
    if not df.empty:
      return df.merge(pres_df,on = 'text' )['mapping'].unique().tolist() # Chỉnh khúc này nếu muốn dùng xác xuất
    else:
      return None


def single_pres(row,pres_df,save_dir):
    temp_df = pd.read_json('/app/public_train/prescription/label/'+row['pres'])
    temp_df = temp_df[temp_df['label']=='drugname']
    mapping = temp_df['mapping'].astype(int).values
    vector = [0]*108
    vector[107] = 1
    for i in mapping:
        vector[i] = 1
    temp_df['text']=temp_df['text'].str.replace(r'^\d+\s*\)\s+','',regex=True).str.lower().apply(lambda x: remove_accents(x))
    for text in temp_df.text:
        ids = findid(text,pres_df)
        for id in ids:
            vector[id] = 1
    result = []
    for pill in row['pill']:
        image_name=pill.replace('json','jpg')
        image_name_raw = pill.replace('.json','')
        image_path = f'/app/public_train/pill/image/{image_name}'
        temp_df = pd.read_json(f'/app/public_train/pill/label/{pill}')
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        for row in temp_df.iloc:
            label = row['label']
        image_count = 0
        for row in temp_df.iloc:
            x1 = row['x']
            y1 = row['y']
            x2 = x1 + row['w']
            y2 = y1 + row['h']
            label = row['label']
            crop_img = img.crop((x1, y1, x2 , y2))
            save_path = f'{save_dir}/{image_name_raw}_{image_count}.jpg'
            image_count+=1
            crop_img.save(save_path)
            result.append((save_path, vector,label))
    return result

def CLASS_DATASET():
    print("RUNNING CLASS TRAIN DATA CREATION")
    save_dir='/app/gen_train_data/image_crop'
    csv_path='/app/gen_train_data/combine_train.csv'
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.read_json('/app/public_train/pill_pres_map.json')
    pres_df = pd.read_csv('/app/pres_df.csv')
    jobs = [
        (row, pres_df, save_dir) for _, row in df.iterrows()
    ]
    result = []
    with Pool() as pool:
        results = pool.starmap(single_pres, tqdm(jobs, total=len(jobs), desc="Creating classification train data"))
        for r in results:
            result.extend(r)
    final_df = pd.DataFrame(result, columns=['pill_path', 'vector','label'])
    final_df.head()
    final_df.to_csv(csv_path)
    print("✅ DONE")
    
    
if __name__=='__main__':
    CREATE_PRESS_DF()
    YOLO_DATASET()
    CLASS_DATASET()