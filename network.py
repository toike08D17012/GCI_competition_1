from torch import nn

INPUT_LENGTH = 7

class Net(nn.Module):
    def __init__(self, n=INPUT_LENGTH):
        super(Net, self).__init__()

        # networkの設定
        self.net = nn.Sequential(
            nn.Linear(n, 32),
            nn.Mish(),
            nn.Linear(32, 16),
            nn.Mish(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
