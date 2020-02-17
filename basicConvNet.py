import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func

import torch.optim as optim

# build data only once
REBUILD_DATA = False


class DogsVsCats:
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    trainingData = []
    catCount = 0
    dogCount = 0

    def makeTrainingData(self):
        for label in self.LABELS:
            # iterate over images
            for f in tqdm(os.listdir(label)):
                try:
                    # join path and filename
                    path = os.path.join(label, f)
                    # load image, grayscale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # img as np array and one hot vector for class
                    self.trainingData.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # just count cats and dogs --> check if balanced
                    if label == self.CATS:
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass
                    # some images might be corrupted

        # shuffle data in place
        np.random.shuffle(self.trainingData)

        np.save("trainingData.npy", self.trainingData)
        print("cats: ", self.catCount)
        print("dogs: ", self.dogCount)


if REBUILD_DATA:
    dogVsCats = DogsVsCats()
    dogVsCats.makeTrainingData()

# just load the data
trainingData = np.load("trainingData.npy", allow_pickle=True)


# print image
# plt.imshow(trainingData[400][0], cmap="gray")
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv2(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.softmax(x, dim=1)


net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in trainingData]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in trainingData])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + BATCH_SIZE]

        optimizer.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))
        predicted_Class = torch.argmax(net_out)
        if predicted_Class == real_class:
            correct += 1
        total += 1

print("acc: ", round(correct/total, 3))
