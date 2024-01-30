import time

import dill
import gymnasium as gym
import numpy as np
import torch
import os

from critic import Critic
from tqc import TQC


class Config():
    env = "Humanoid-v4"
    env_name = "humanoid_tqc"
    total_step=2e6
    num_test = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

para = Config()
model = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
env = gym.make(para.env, render_mode="human")
state_dim = env.observation_space.shape[0]
critic_net = Critic(state_dim=state_dim, device=para.device, hidden_dim=[512, 256, 128, 32])

loss_list = list()
for i in range(para.num_test):    
    state = env.reset()[0]
    episode_reward = 0
    loss_list.clear()
    for _ in range(1000):
        action, _state = model.predict(state, deterministic=True)

        # env.render()         
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward

        loss = critic_net.learn(state, reward)
        print(loss)
        loss_list.append(loss)
        # loss = critic_net.learn()
        if done:
            break
# print(loss_list)
    print(i, np.mean(loss_list))
if not os.path.exists('save_critic/'):
    os.makedirs('save_critic/')
savepath = "save_critic/{}_{}.pkl".format(para.env_name, para.total_step)
torch.save(critic_net.state_dict(), savepath)
            
        # print("final reward = ", np.mean(episode_rewards), "±", np.std(episode_rewards))