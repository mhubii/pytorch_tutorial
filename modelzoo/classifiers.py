import torch
import torch.nn as nn

class ConvClassifier(nn.Module):
    def __init__(self, c, h, w):
        super(ConvClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Flatten()
        )

        n = self._numel_features([1,c,h,w])  # BxCxHxW

        self.classifcation = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1)
        )

    def _numel_features(self, shape):
        x = torch.rand(shape)
        x = self.features(x)
        return x.numel()

    def forward(self, img):
        out = self.features(img)
        out = self.classifcation(out)
        return out


if __name__ == '__main__':
    b,c,h,w = 1,3,64,64
    img = torch.rand([b,c,h,w])

    model = ConvClassifier(c,h,w)
    log_prob = model(img)
    print(torch.exp(log_prob))
