import os
import random
from glob import glob

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


def YOLO_DATASET():
    file_path = '/app/public_train/pill/label/*.json'
    save_dir='/app/gen_train_data/yolo_pill'
    split_ratio=0.1

    print("RUNNING YOLO TRAIN DATA CREATION")
    print("-"*10)
    print('save to:',save_dir)
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

    for i in tqdm(range(train_len)):
        file = files[i]
        img_path = file.replace('label', 'image').replace('.json', '.jpg')
        img = Image.open(img_path)
        try:
            img = ImageOps.exif_transpose(img)
        except:
            pass
        W,H = img.size
        new_width = 640
        new_height = 640
        img = img.resize((new_width, new_height), Image.LANCZOS)
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
    print("✅ RUNNING YOLO TRAIN DATA CREATION")



def CLASS_DATASET():
    save_dir='/app/gen_train_data/image_crop'
    file_paths='/app/public_train/pill/images/*.jpg'
    csv_path='/app/combine_train.csv'
    print("RUNNING CLASSIFICATION TRAIN DATA CREATION")
    print("-"*10)
    print('save to:',save_dir)
    files=glob(file_paths)

if __name__ == '__main__':
    YOLO_DATASET()
    # CLASS_DATASET()