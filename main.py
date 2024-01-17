from sac_agent import Agent
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import multiprocessing
import dill
from dynamicsynapse import DynamicSynapse
from torchvision import transforms as T
from Adapter.RangeAdapter import RangeAdapter
import copy

np.set_printoptions(threshold=np.inf)
# range_adapter = RangeAdapter()
# range_adapter.init_recording()

def preprocessing(data):
    # torch.abs(data)
    for li in data:
        li = (li - torch.mean(li)) / torch.std(li)
    return data

class Parameters():
    def __init__(self) -> None:
        self.k1 = 0.1
        self.k2 = 0.002
        self.max_episode_num = 5000
        self.is_train = False
        self.is_continue_train = True
        self.continue_train_episodes = 3000
        self.modelfilepath = "humanoid_v4_sac_2975646.pth"
        self.dt = 15


# parser = argparse.ArgumentParser(description="mode")
# parser.add_argument("--istrain","-t", type=int, default=0, nargs="?")
# parser.add_argument("--iscotinuetrain","-c", type=int, default=0, nargs="?")
# parser.add_argument("--continuetrainepisodes","-n", type=int, default=10000, nargs="?")
# args = parser.parse_args()

para = Parameters()

max_episode_num = para.max_episode_num
env = gym.make('Humanoid-v4', render_mode="human")
GPU_NUM = 0
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

is_train = para.is_train
is_cotinue_train = para.is_continue_train
continue_train_episodes = para.continue_train_episodes
# print(is_train)
# is_train = True

if is_train:
    agent = Agent(env, device)
    agent.train(max_episode_num=max_episode_num, max_time=1000)

else:
    agent = Agent(env, device, write_mode=True)
    state_dict = torch.load(agent.save_model_path + para.modelfilepath)
    agent.load_state_dict(state_dict)
    print("*****************Start Continual Training****************************")
    i = 0
    flag = 1


    avg_episode_reward = 0.
    num_test = 5
    episode_rewards = []


    with open("gradient_mean.pkl", "rb") as f:
        grandient_mean = dill.load(f)

    with open("gradient_max.pkl", "rb") as f:
        grandient_max = dill.load(f)

    with open("gradient_mean_abs.pkl", "rb") as f:
        grandient_mean_abs = dill.load(f)

    # gm = torch.tensor(grandient_mean)
    # gm = torch.tensor([item.detach().numpy() for item in grandient_mean])

    gm = preprocessing(grandient_mean_abs)

    # gm = torch.tensor(gm)

    with open("weight_mean.pkl", "rb") as f1:
        weight = dill.load(f1)

    # w = [j.cpu().numpy() for j in weight]
    # w = preprocessing(weight)
    gm = [torch.abs(g * para.k1) for g in gm]
    weight = [torch.abs(w * para.k2) for w in weight]
    amp_init = [gm[i] + weight[i] for i in range(len(gm))]

    trace_reward = list()
    if is_cotinue_train == 0:
        for _ in range(num_test):    
            state = env.reset()[0]
            episode_reward = 0
    
            for _ in range(1000):
                action = agent.actor.get_action(torch.tensor([state]).to(device).float(), stochastic=False)

                # env.render()         
                state, reward, done, _, _ = env.step(action[0])
                episode_reward += reward
                if done:
                    break 
            episode_rewards.append(episode_reward)

        # print('avg_episode_reward :', np.mean(episode_rewards))
    
    elif is_cotinue_train:

        
        agent.actor.optimizer_dynamic = DynamicSynapse(agent.actor.parameters(), lr=agent.actor.lr, amp=amp_init, period=4000, dt=para.dt)
        for episode_idx in range(continue_train_episodes):
            state = env.reset()[0]
            episode_reward = 0
            step = 0
            while True:  
                step += 1
                action = agent.actor.get_action(torch.tensor([state]).to(device).float(), stochastic=False)
                # print(action)
                state, reward, done, _, _ = env.step(action[0])
                # print(done)

                # reward = range_adapter.step_dynamics(para.dt, reward)
                trace_reward.append(reward)
                # range_adapter.recording()
                # range_adapter.update()

                agent.actor.learn_dynamic(reward)
                episode_reward += reward
                agent.total_step += 1

                # if step == 10000:
                #     plt.plot(range(10000), trace_reward, "g-")
                #     plt.savefig("range_adapter.png")
                #     plt.show()

                if done:
                    # print("break")
                    break 
            episode_rewards.append(episode_reward)
            if (episode_idx + 1) % agent.print_period == 0:
                content = 'Tot_step: {} \t | Episode: {} \t  | Reward : {} \t '.format(agent.total_step, episode_idx + 1, np.mean(episode_rewards[-agent.print_period:]))
                agent.my_print(content)

                if agent.write_mode:
                    agent.writer.add_scalar('Reward/reward', np.mean(episode_rewards[-agent.print_period:]), agent.total_step)
                    

            
        

        