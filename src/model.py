import numpy as np
import torch
from torch import nn

from dataset import int_dict


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.channels = [(1, 32), (32, 64), (64, 128), (128, 128), (128, 256)]
        self.kernels = [(5, 5), (5, 5), (3, 3), (3, 3), (3, 3)]
        self.paddings = [2, 2, 1, 1, 1]
        self.pooling_kernels = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]

        self.layers = []
        for (in_channels, out_channels), kernel, padding, pooling_kernel in zip(
            self.channels, self.kernels, self.paddings, self.pooling_kernels
        ):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pooling_kernel),
                )
            )

        self.inner_layers = self.layers[1:]
        self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = self.layers

    def forward(self, x):
        out = self.layers[0](x)

        for layer in self.inner_layers:
            out = layer(out)

        out = out.squeeze(dim=2).transpose(1, 2)
        return out


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.atrous_conv = nn.Conv2d(512, 80, kernel_size=1, dilation=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.atrous_conv(out.permute(0, 2, 1).unsqueeze(3))

        out = out.squeeze(3).permute((2, 0, 1))
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = RNN()

    def forward(self, x):
        out = self.cnn(x)
        out = self.rnn(out)
        return out

    @staticmethod
    def decode(out):
        with torch.no_grad():
            softmax_out = out.softmax(2).argmax(2).permute(1, 0).cpu().numpy()
            words = []
            for i in range(softmax_out.shape[0]):
                without_duplicates = softmax_out[i, :][
                    np.insert(np.diff(softmax_out[i, :]).astype(np.bool), 0, True)
                ]
                int_values = without_duplicates[without_duplicates != 0]
                words.append(
                    "".join([int_dict[value] for value in int_values.astype(int)])
                )

        return words


if __name__ == "__main__":
    model = Model()
    print(model)
