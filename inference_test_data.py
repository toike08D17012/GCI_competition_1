from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier

from preprocess import preprocess, fix_seed
from algorithm import Algorithm

input_dir = Path('./data')
result_dir = Path('./result_test_data')

save_path = result_dir / 'result_20220604_2.csv'

seed = 1234

epoch = 2000
batch_size = 256


def main():
    assert not save_path.exists(), 'This file name is already used. Please change save name.'

    result_dir.mkdir(exist_ok=True)

    train_data_path = input_dir / 'train.csv'
    test_data_path = input_dir / 'test.csv'

    # seedの固定
    fix_seed(seed)

    train_data, test_data = preprocess(train_data_path, test_data_path)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_test = test_data

    model = Algorithm(epoch, batch_size,)
    # model = GradientBoostingClassifier(random_state=0)

    # model.fit(X_train, y_train)
    loss_list = model.train(X_train, y_train)
    plt.figure()
    plt.plot(loss_list)
    plt.savefig(result_dir / 'loss_inference_test_data.png')

    result = {'PassengerId': [], 'Perished': []}
    number = 892
    for data in X_test:
        result['PassengerId'].append(number)
        # result['Perished'].append(model.inference(data[1:])[0])
        if any(np.isnan(data)):
            print(data)
        result['Perished'].append(model.inference(data)[0])
        # result['Perished'].append(int(model.predict(data.reshape(1, -1))[0]))

        number += 1
        
    df = pd.DataFrame(result)

    

    df.to_csv(save_path)

if __name__ == '__main__':
    main()