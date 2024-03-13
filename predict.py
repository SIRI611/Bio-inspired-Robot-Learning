import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Predict(nn.Module):
    def __init__(self, batch_size, device, gamma=0.99, lr=1e-5):
        super(Predict, self).__init__()
        # self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.conv1 = nn.Conv2d(1, 6, (5, 11))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*23*12, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.activation = nn.ReLU()
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # self.apply(self.weights_init_)

    
    # def weights_init_(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight, gain=1)
    #         torch.nn.init.constant_(m.bias, 0)


    def forward(self, state):
        
        x = torch.Tensor(state)
        x = nn.functional.max_pool2d(self.activation(self.conv1(x)), (1, 4))
        x = nn.functional.max_pool2d(self.activation(self.conv2(x)), (1, 4))
        x = x.reshape((x.size()[0], -1))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x

    def learn(self, states, td_target_values):

        current_value = self.forward(states)
        loss = self.criterion(current_value, td_target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

