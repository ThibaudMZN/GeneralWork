#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def loadData():
    im = Image.open("./data/digits.png")
    arr = np.array(im) / 255 * 2 - 1
    gray = np.reshape(arr, (1000, 2000))
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    x = np.array(cells)

    train = x[:, :50].reshape(2500, 20, 20).astype(np.float32)
    test = x[:, 50:100].reshape(2500, 20, 20).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis].astype(np.int)
    test_labels = train_labels.copy()

    train = torch.Tensor(train)
    train = train.unsqueeze(1)  # for Grayscale squeezing
    train_labels = torch.LongTensor(train_labels)

    train_data = torch.utils.data.TensorDataset(
        train, train_labels.view(-1))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=2, shuffle=True)

    test = torch.Tensor(test)
    test = test.unsqueeze(1)  # for Grayscale squeezing
    test_labels = torch.LongTensor(test_labels)

    test_data = torch.utils.data.TensorDataset(
        test, test_labels.view(-1))
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=2, shuffle=True)

    return train_loader, test_loader  # , test, train_labels, test_labels


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, nepoch, trainloader, testloader):
        for epoch in range(nepoch):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                if i % 300 == 299:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 300))
                    running_loss = 0.0
            acc = self.accuracy_test(testloader)
            print("Accuracy at epoch %d : %d %%" % (epoch + 1, acc))
        print('Finished Training')

    def accuracy_test(self, testloader):
        correct = 0
        total = 0
        for data in testloader:
            im, labels = data
            outputs = self(Variable(im))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        return (100 * correct / total)


if __name__ == "__main__":

    trainloader, testloader = loadData()

    net = Net()
    net.train(10, trainloader, testloader)
