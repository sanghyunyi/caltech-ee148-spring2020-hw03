import torch
import torch.nn as nn
import torch.nn.functional as F

class Net0(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5,5), stride=1)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net1(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net2(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=(3,3), stride=1)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net3(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=(3,3), stride=1)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net4(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=(3,3), stride=1)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.avg_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net5(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=(3,3), stride=1)
        p = .8
        self.dropout1 = nn.Dropout2d(p)
        self.dropout2 = nn.Dropout2d(p)
        self.dropout3 = nn.Dropout2d(p)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

class Net6(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=(3,3), stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.bn3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

