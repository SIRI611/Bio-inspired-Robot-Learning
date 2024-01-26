import gymnasium as gym
import dill
from tqc import TQC
import torch
import numpy as np
import time
from critic import Critic



class Config():
    env = "Walker2d-v4"
    env_name = "walker2d_tqc"
    total_step=1e6
    num_test = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    

para = Config()
model = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
env = gym.make(para.env, render_mode="human")
state_dim = env.observation_space.shape[0]
critic_net = Critic(state_dim=state_dim, device=para.device, hidden_dim=[8, 4])

loss_list = list()
for _ in range(para.num_test):    
    state = env.reset()[0]
    episode_reward = 0

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
savepath = "save_critic/{}_{}.pkl".format(para.env_name, para.total_step)
torch.save(critic_net.state_dict(), savepath)
            
        # print("final reward = ", np.mean(episode_rewards), "±", np.std(episode_rewards))