import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=[256, 256], gamma=0.99, lr=1e-4):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.device = device
        
        self.dim_list = hidden_dim + [1]

        self.first_layer = nn.Linear(in_features=state_dim+action_dim, out_features=hidden_dim[0])

        self.layer_module = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self.dim_list[:-1], self.dim_list[1:])]
        )
        
        self.activation = nn.ReLU()


        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.apply(self.weights_init_)

    
    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)


    def forward(self, state, action):
        
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.first_layer(x))
        
        for layer in self.layer_module[:-1]: # not include out layer
            x = self.activation(layer(x))

        x = self.layer_module[-1](x)

        return x

    def learn(self, states, actions, td_target_values):

        current_value = self.forward(states, actions)
        loss = torch.mean((td_target_values - current_value)**2)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
