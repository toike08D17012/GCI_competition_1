import sys
from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from preprocess import preprocess, fix_seed
from postprocess import evaluate
from algorithm import Algorithm

epoch = 5000
batch_size = 256

seed = 1234

columns = ['TP', 'FP', 'TN', 'FN', 'recall', 'precision', 'accuracy', 'f1_score', ]

model_list = ['dnn', ]


def main():
    # GPUかCPUかを自動設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dir = Path('./data')
    result_dir = Path('./result')
    result_dir.mkdir(exist_ok=True)

    train_data = pd.read_csv(input_dir / 'train.csv')

    # seedの固定
    fix_seed(seed)

    # preprocess
    data, _ = preprocess(train_data)

    # train_test_split
    train_data, test_data = train_test_split(data, random_state=0, test_size=0.1)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    model_dict = {
        'dnn': Algorithm(epoch, batch_size),
        'gnb': GaussianNB(),
        'lr': LogisticRegression(max_iter=1000000),
        'svm': LinearSVC(max_iter=1000000),
        'gbdt': GradientBoostingClassifier(random_state=0)
        }

    for model_name, model in model_dict.items():
        if model_name == 'dnn':
            loss_list = model.train(X_train, y_train)
            plt.figure()
            plt.plot(loss_list)
            plt.savefig(result_dir / 'loss.png')
        else:
            model.fit(X_train,y_train)

    result = pd.DataFrame(columns=columns)

    for model_name, model in model_dict.items():
        if model_name == 'dnn':
            y_predict = model.inference(X_test)
        else:
            y_predict = model.predict(X_test)

        df_temp = evaluate(y_predict, y_test, model_name)

        result = pd.concat([result, df_temp])

        result.to_excel(result_dir / 'result.xlsx')


if __name__ == '__main__':
    main()
