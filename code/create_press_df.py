from glob import glob
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    files = glob('/app/public_train/prescription/label/*.json')
    pres_df = pd.DataFrame()
    for file in tqdm(files):
        temp_df = pd.read_json(file)
        temp_df['file_name'] = file.split('\\')[-1]
        pres_df = pd.concat([pres_df,temp_df])
    pres_df = pres_df[pres_df['label']=='drugname']
    pres_df['mapping'] = pres_df['mapping'].astype(int)
    pres_df['text']=pres_df['text'].str.replace(r'^\d+\s*\)\s+','',regex=True).str.lower().apply(lambda x: remove_accents(x))
    pres_df.drop(['id'], axis=1, inplace=True)
    pres_df.head()
    pres_df.to_csv('/app/pres_df.csv')
