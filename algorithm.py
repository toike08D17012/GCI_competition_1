import time
from pathlib import Path

import matplotlib.pyplot as plt
from progressbar import ProgressBar
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch import optim

from network import Net
from trainer import Trainer

result_dir = Path('./result')


class Algorithm:
    def __init__(self, epoch=5000, batch_size=100, initialize='initialize', ckpt_path=None):
        self.epoch = epoch
        self.batch_size = batch_size

        # GPUかCPUかを自動設定
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Networkのインスタンス化
        self.model = Net().to(self.device)
        # 最適化手法の決定
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        # Lossの設定
        self.criterion = nn.MSELoss()

        # checkpointパスの設定
        self.checkpoint_path = Path('./checkpoints')
        self.checkpoint_path.mkdir(exist_ok=True)

        # resume
        if initialize == 'resume':
            if ckpt_path is None:
                ckpt_path = self.checkpoint_path / 'checkpoint.cpt'

            self._resume(ckpt_path)

    
    def _resume(self, ckpt_path):
        # Load the model
        ckpt = torch.load(ckpt_path)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['opt_state_dict'])


    def _make_dataset(self, X, y):
        # Tensorに変更
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape([-1, 1])

        # Datasetを作成
        Dataset = torch.utils.data.TensorDataset(X, y)

        Loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last = True)

        return Loader


    def train(self, X_train, y_train):
        self.model.train()
        trainer = Trainer(self.model, self.optimizer, self.criterion)

        train_loader = self._make_dataset(X_train, y_train)

        loss_list = []

        start_ = time.time()
        for ite in range(self.epoch):
            start = time.time()
            for X, y in train_loader:
                # GPUメモリにデータを移動
                X = X.to(self.device)
                y = y.to(self.device)

                # 実際にフィッティングし，勾配を更新
                loss = trainer.fit(X, y)
                loss = loss.to('cpu').detach().numpy().copy()

            loss_list.append(loss)

            elapsed_time = time.time() - start
            elapsed_time_total = time.time() - start_
            print(f'iteration: {ite + 1}, loss: {loss:5f}, elpsed time(epoch): {elapsed_time:5f}, elpsed time(total): {elapsed_time_total:5f}')

            if (ite + 1) % 100 == 0:
                # checkpointの設定
                checkpoint = {
                    'epoch': ite,
                    'model_state_dict': self.model.state_dict(),
                    'opt_state_dict': self.optimizer.state_dict(),
                    'loss': loss_list
                }
                torch.save(checkpoint, self.checkpoint_path / 'checkpoint.cpt')
                torch.save(checkpoint, self.checkpoint_path / f'checkpoint_{ite+1}.cpt')

                plt.plot(loss_list)
                plt.savefig(result_dir / f'loss_{ite+1}.png')

            if loss < 0.075:
                checkpoint = {
                    'epoch': ite,
                    'model_state_dict': self.model.state_dict(),
                    'opt_state_dict': self.optimizer.state_dict(),
                    'loss': loss_list
                }
                torch.save(checkpoint, self.checkpoint_path / 'checkpoint.cpt')

                plt.plot(loss_list)
                plt.savefig(result_dir / f'loss_{ite+1}.png')
                plt.clf()
                break

        return loss_list


    def inference(self, X_test, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.checkpoint_path / 'checkpoint.cpt'

        self._resume(ckpt_path)

        # 実際に推論
        self.model.eval()

        X = torch.from_numpy(X_test).float().to(self.device)

        with torch.no_grad():
            y_predict = self.model.forward(X)

        y_predict = y_predict.to('cpu').detach().numpy().copy()
        y = []
        for predict in y_predict:
            if predict >= 0.5:
                y.append(1)
            else:
                y.append(0)

        return y       
