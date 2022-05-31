from pathlib import Path

from torch import nn


class Trainer(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
    ):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.save_path = Path('./checkpoints')
        self.save_path.mkdir(exist_ok=True)

    def fit(self, x, t):
        """
        x: 入力
        t: 正解
        """
        # 勾配を0にリセット
        self.optimizer.zero_grad()

        # 推定値
        y = self.model.forward(x)

        loss = self.criterion(y, t)
        loss.backward()
        self.optimizer.step()

        return loss