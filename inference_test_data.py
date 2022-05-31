from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from preprocess import preprocess, fix_seed
from algorithm import Algorithm

input_dir = Path('./data')
result_dir = Path('./result_test_data')

seed = 1234

epoch = 500
batch_size = 256


def main():
    result_dir.mkdir(exist_ok=True)

    train_data = pd.read_csv(input_dir / 'train.csv')
    test_data = pd.read_csv(input_dir / 'test.csv')

    # seedの固定
    fix_seed(seed)

    # GPUかCPUかを自動設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, test_data = preprocess(train_data, test_data)
    test_data = test_data[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']].values

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_test = test_data

    model = Algorithm(epoch, batch_size)


    loss_list = model.train(X_train, y_train)
    plt.figure()
    plt.plot(loss_list)
    plt.savefig(result_dir / 'loss_inference_test_data.png')

    result = {'PassengerId': [], 'Perished': []}
    for data in X_test:
        result['PassengerId'].append(data[0])
        result['Perished'].append(model.inference(data[1:]))
        
    df = pd.DataFrame(result)

    df.to_csv(result_dir / 'result_20220531.csv')

if __name__ == '__main__':
    main()