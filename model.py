import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class HousePriceModel(nn.Module):
    def __init__(self, D_in, D_out=1):
        super(HousePriceModel, self).__init__()
        self.D_in = D_in
        self.hidden1 = 256
        self.hidden2 = 128
        self.hidden3 = 64
        self.hidden4 = 32
        self.D_out = 1
        self.fc1 = nn.Linear(self.D_in, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.fc4 = nn.Linear(self.hidden3, self.hidden4)
        self.out = nn.Linear(self.hidden4, self.D_out)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        output = F.relu(self.out(output))
        return output

