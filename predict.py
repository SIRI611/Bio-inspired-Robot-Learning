import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Predict(nn.Module):
    def __init__(self, batch_size, device, gamma=0.99, lr=3e-4):
        super(Predict, self).__init__()
        # self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*10*197, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)
        
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
        x = self.conv1(x)
        x = self.activation(self.conv2(x))
        x = nn.functional.max_pool2d(self.activation(x), (2, 2))
        x = x.reshape((x.size()[0], -1))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        return x

    def learn(self, states, td_target_values):

        current_value = self.forward(states)
        loss = self.criterion(current_value, td_target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

