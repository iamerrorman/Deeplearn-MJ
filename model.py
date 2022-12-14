from torch import nn
import torch


class BaseLay1(nn.Module):
    def __init__(self, in_channels):
        super(BaseLay1, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, intputs):
        return self.lay(intputs) + intputs


class BaseLay2(nn.Module):
    def __init__(self, in_channels):
        super(BaseLay2, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, intputs):
        return self.lay(intputs) + intputs


class Model_v1(nn.Module):
    def __init__(self, in_channels, targets):
        super(Model_v1, self).__init__()
        base_lay1 = BaseLay1(128)
        base_lay2 = BaseLay2(128)
        self.x1_lay1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=(1, 0), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        lays = []
        for i in range(4):
            lays.append(base_lay1)
        self.x1_lay2 = nn.Sequential(*lays)

        self.x2_lay1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        lays = []
        for i in range(4):
            lays.append(base_lay2)
        self.x2_lay2 = nn.Sequential(*lays)

        lays = []
        for i in range(8):
            lays.append(base_lay1)
        self.lay = nn.Sequential(*lays)
        self.flatten = nn.Flatten()
        self.cf = nn.Sequential(
            nn.Linear(4 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, targets)
        )

    def cul(self, x1, x2):
        x1 = self.x1_lay1(x1)
        x1 = self.x1_lay2(x1)
        x2 = self.x2_lay1(x2)
        x2 = self.x2_lay2(x2)
        # print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=2)
        x = self.lay(x)
        x = self.flatten(x)
        x = self.cf(x)
        return x

    def forward(self, x1, x2):
        return self.cul(x1, x2)

    def predict(self, x1, x2, mask):
        x = self.cul(x1, x2)
        for idx, item in enumerate(mask):
            x[idx][item == 0] = -100.0
        return x


if __name__ == '__main__':
    x1 = torch.randint(2, (2, 71, 3, 9), dtype=torch.float)
    x2 = torch.randint(2, (2, 71, 1, 7), dtype=torch.float)
    mask = torch.randint(2, (1, 35))
    model = Model_v1(71, 35)
    # print(model)
    print(model.predict(x1, x2, mask).argmax(1))

