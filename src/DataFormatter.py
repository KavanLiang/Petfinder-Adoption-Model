import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from pandas.io.json import json_normalize
import json
from glob import glob
import random

import pandas as pd
import pickle

from tqdm import tqdm


def mkdir(file_path):
    """
    Make a new directory with the name file_path
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def readSentiments(fn):
    file = 'SLink/PetfinderFiles/train_sentiment/' + fn['PetID'] + '.json'
    if os.path.exists(file):
        with open(file, encoding='utf-8') as data_file:
            data = json.load(data_file)
        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score'])
    else:
        return pd.Series([0,0],index=['mag','score'])

def build_train_val_directories():
    """
    Builds training and validation directories.
    Returns a dictionary containing class weights.
    """
    for i in range(5):
        mkdir(f'SLink/PetfinderDatasets/Train/{i}/')
        mkdir(f'SLink/PetfinderDatasets/Val/{i}/')

    df = pd.read_csv('SLink/PetfinderFiles/train/train.csv')

    df['NameLength'] = df['Name'].str.len()
    df['DescriptionLength'] = df['Description'].str.len()

    df = df.drop(columns=['Name', 'RescuerID', 'Description'])
    df[['SentimentMagnitude', 'SentimentScore']] = df[['PetID']].apply(lambda x: readSentiments(x), axis=1)

    df = df.set_index('PetID')

    # onehot encode categorical data
    df['State'] = df['State'].replace({41336: 0,
                                       41325: 1,
                                       41367: 2,
                                       41401: 3,
                                       41415: 4,
                                       41324: 5,
                                       41332: 6,
                                       41335: 7,
                                       41330: 8,
                                       41380: 9,
                                       41327: 10,
                                       41345: 11,
                                       41342: 12,
                                       41326: 13,
                                       41361: 14})
    df.fillna(0, inplace=True)

    onehot_df = df.drop(columns=['Age', 'Quantity', 'Fee', 'PhotoAmt', 'AdoptionSpeed', 'VideoAmt', 'NameLength', 'DescriptionLength', 'SentimentScore', 'SentimentMagnitude'])
    continuous_df = df[['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt', 'NameLength', 'DescriptionLength', 'SentimentScore', 'SentimentMagnitude']]

    for col in onehot_df.columns:
        onehot_df = pd.concat([onehot_df, pd.get_dummies(df[col], prefix=col)], axis=1)
        onehot_df.drop([col], axis=1, inplace=True)

    class_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    num_training = 0
    num_val = 0

    for index, row in tqdm(continuous_df.iterrows()):
        label = int(df["AdoptionSpeed"][index])
        data = [np.array(onehot_df.loc[index]), np.array(row)]
        if data[0].shape == (369,) and data[1].shape == (9,):
            if row['PhotoAmt'] == 0:
                img = np.ones(shape=(200, 200, 3))
                data = data + [img]
                if random.random() > 0.25:
                    pickle.dump(data,
                                open(f'SLink/PetfinderDatasets/Train/{label}/{index}-0', 'wb'))
                    num_training += 1
                    class_count[label] += 1
                else:
                    pickle.dump(data,
                                open(f'SLink/PetfinderDatasets/Val/{label}/{index}-0', 'wb'))
                    num_val += 1
            else:
                for i in range(1, int(row['PhotoAmt'] + 1)):
                    img = img_to_array(load_img(f'SLink/PetfinderFiles/train_images/{index}-{i}.jpg'))
                    data_to_write = data + [img]
                    if random.random() > 0.25:
                        pickle.dump(data_to_write, open(
                            f'SLink/PetfinderDatasets/Train/{label}/{index}-{i}', 'wb'))
                        num_training += 1
                        class_count[label] += 1
                    else:
                        pickle.dump(data_to_write,
                                    open(f'SLink/PetfinderDatasets/Val/{label}/{index}-{i}',
                                         'wb'))
                        num_val += 1
    max_count = max(class_count.values())
    return {k: max_count / v for k, v in class_count.items()}, num_training, num_val

if __name__ == '__main__':
    print(build_train_val_directories())
