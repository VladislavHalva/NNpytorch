import numpy as np

import torch
from torch import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets

### Dataset settings ###

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


### NN Definition ###

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


X = torch.rand((28, 28))
X = X.view(-1, 28 * 28)

net = Net()
output = net(X)

optimizer = optim.Adam(net.parameters(), lr=0.001)

# iterations through full dataset
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainSet:
        X, y = data
        # clear gradients
        net.zero_grad()
        # forward pass
        output = net(X.view(-1, 28 * 28))
        # calc loss
        loss = F.nll_loss(output, y)
        # backpropagation
        loss.backward()
        # adjust weights
        optimizer.step()
    print(loss)

# test accuracy
correct = 0
total = 0

with torch.no_grad():
    for data in trainSet:
        X, y = data
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("accuracy: ", correct / total * 100)
