
import numpy as np
import torch
import torch.nn as nn


class Concat_2(nn.Module):
    """
    This class is used to concatenate 2 matrix on the second dimension.The matrix must have 3 dimensions.
    """

    def __init__(self):
        super(Concat_2, self).__init__()

    def forward(self, matrix_1, matrix_2):
        tensor_concat = torch.from_numpy(
            np.full((matrix_1.size(0), matrix_1.size(1) + matrix_2.size(1), matrix_2.size(2)), None, dtype='float'))
        for i in range(matrix_1.size(0)):
            tensor_concat[i] = torch.cat([matrix_1[i], matrix_2[i]], dim=0)

        return tensor_concat.contiguous().float()


class View(nn.Module):
    def __init__(self, x_size, y_size, z_size):
        super(View, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def forward(self, input):
        return input.view((self.x_size, self.y_size, self.z_size))


class ResidualBlock(nn.Module):
    
    def __init__(self, n_channel_input, n_channel_output, kernel_size, stride, dilation, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_channel_input, n_channel_output, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)

        self.batch_norm1 = nn.BatchNorm1d(n_channel_output)
        self.relu1 = nn.LeakyReLU(0.9)
        self.drop = nn.Dropout(0.5)

        self.net = nn.Sequential(self.conv1, self.batch_norm1, self.relu1, self.drop)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, n_layers, n_inputs, n_outputs, dilation, receptive_field, kernel_size, stride, device):
        super(Encoder, self).__init__()
        layers = []

        for i in range(n_layers):
            layer = ResidualBlock(n_inputs if i == 0 else n_outputs[i - 1], n_outputs[i], kernel_size, stride,
                                  dilation[i], 0)

            layers += [layer]

        self.net = nn.Sequential(*layers)
        self.device = device

    def forward(self, X, y):
        self.concat = Concat_2()
        return self.net(self.concat(X, y).to(self.device)).to(self.device)

class Decoder(nn.Module):
    def __init__(self, n_channel, n_channel_X, future_size, device):
        super(Decoder, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)

        self.dense1 = nn.Linear(future_size * n_channel_X, n_channel * future_size)

        self.batch1 = nn.BatchNorm1d(n_channel * future_size)
        self.relu = nn.LeakyReLU(0.9)
        self.dense2 = nn.Linear(n_channel * future_size, n_channel_X)
        self.batch2 = nn.BatchNorm1d(n_channel_X)

        self.reshap = View(-1, n_channel_X, 1)

        self.net = nn.Sequential(self.flatten, self.dense1, self.batch1
                                 , self.relu, self.dense2, self.batch2, self.reshap)
        self.relu2 = nn.LeakyReLU(0.9)
        self.device = device
        self.f = future_size

        self.flatten = nn.Flatten(start_dim=1)
        self.final_linear = nn.Linear(n_channel_X, 1)
        self.reshap2 = View(-1, 1, 1)

        self.net2 = nn.Sequential(self.relu2, self.flatten, self.final_linear, self.reshap2)

    def forward(self, info, memory):
        out1 = self.net(info).to(self.device)
        out2 = self.net2((out1 + memory).float()).to(self.device)
        return out2
