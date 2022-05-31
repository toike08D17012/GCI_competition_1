import os
import random

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def substitute(df):
    df['Sex'] = [1 if i == 'male' else 0 for i in df['Sex']]

    # Embarked: S = 2 C = 1 Q = 0, NaN = 5
    temp = df['Embarked']
    temp_list = []
    for i in temp:
        if i == 'S':
            temp_list.append(2)
        elif i == 'C':
            temp_list.append(1)
        elif i == 'Q':
            temp_list.append(0)
        else:
            temp_list.append(2)
    df['Embarked'] = temp_list

    # if Age == NaN -> Age = Age.mean()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    return df

def preprocess(df, test_df=None):
    df = substitute(df)

    scaler = MinMaxScaler()
    ret = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Perished']]
    # Scaling to 0~1.
    scaler.fit(ret)
    ret = scaler.transform(ret)

    if test_df is not None:
        test_df = substitute(test_df)

        scaler = MinMaxScaler()
        scaler.fit(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']])

        test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']] = \
            scaler.transform(test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']])

    return ret, test_df


def fix_seed(seed=1234):
    """ シードの固定
    Args:
        seed (int): A random seed. default to 1234.
    """    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str('1234')
    np.random.seed(seed)
    torch.manual_seed(seed)
