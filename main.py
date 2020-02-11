import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    data_x = np.asarray([1, 2, 2.5, 2.7, 3, 3.7, 4.0, 4.2, 4.4, 5, 6, 7],
                        dtype=np.float32) / 7
    data_y = np.asarray([1.3, 1.5, 4.0, 3.0, 3.9, 4.9, 4.3, 5.8, 5.2, 5.2, 5.5, 5.4],
                        dtype=np.float32) / 7
    x = plt.plot(data_x, data_y, marker='o', color='r', ls='')

    torch_x = torch.from_numpy(data_x)
    torch_y = torch.from_numpy(data_y)

    z = torch.FloatTensor(2)
    z[...] = 0

    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    odd = arr[::1]
    print(odd)


main()


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
