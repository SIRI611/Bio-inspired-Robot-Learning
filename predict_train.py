import time

import dill
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import os

from critic import Critic
from tqc import TQC
from collections import deque
from scipy.signal import savgol_filter


class Config():
    env = "Humanoid-v4"
    env_name = "humanoid_tqc"
    total_step=2e6
    num_test = 5000
    if_train = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def f2(x):
    return 2 / (1 + np.exp(- x * 0.02)) - 1

para = Config()
model = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
# env = gym.make(para.env, render_mode="human")
env = gym.make(para.env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
critic_net = Critic(state_dim=state_dim+action_dim, device=para.device, hidden_dim=[256, 128, 64, 16])


if para.if_train:
    num = int(1e6)
    replay_buffer_len = int(1e6)
    replay_buffer_list = [i for i in range(replay_buffer_len)]
    replay_buffer = deque(maxlen=replay_buffer_len)
    fall_step = deque(maxlen=replay_buffer_len)
    loss_list = list()

    action_list = deque(maxlen=num)
    state_list = deque(maxlen=num)
    state_action_list = deque(maxlen=num)
    nowtime = time.strftime("%m%d_%H%M%S", time.localtime())
    for i in range(para.num_test):    
        state = env.reset()[0]
        episode_reward = 0
        loss_list.clear()
        state_action_list.clear()
        step = 0
        while(True):
            step += 1 
            action, _state = model.predict(state, deterministic=True)

            state, reward, done, _, _ = env.step(action)
            state_action_list.append(np.concatenate((state, action)))
            
            if done:
                iter = min(num, step)
                replay_buffer.extend(state_action_list)
                fall_step.extend([iter - m - 1 for m in range(iter)])

                repeat_num = min(50, step)
                replay_buffer_temp = np.repeat([state_action_list[-m-1] for m in range(repeat_num)], int((step) / repeat_num), axis=0)
                fall_step_temp = np.repeat([n for n in range(repeat_num)], int((step) / repeat_num), axis=0)
                # print(len(replay_buffer_temp), len(fall_step_temp))

                replay_buffer.extend(replay_buffer_temp)
                fall_step.extend(fall_step_temp)

                if i > int(para.num_test / 100) or len(replay_buffer) == replay_buffer_len:
                    for _ in range(5):
                        if len(replay_buffer) < replay_buffer_len:
                            sample = np.random.choice([i for i in range(len(replay_buffer))], size=512, replace=False)
                        else:
                            sample = np.random.choice(replay_buffer_list, size=1024, replace=False)
                        for j in sample:
                            loss = critic_net.learn(replay_buffer[j], f2(fall_step[j]))
                            loss_list.append(loss)
                break
        if len(loss_list) > 0:
            max_loss = np.max(loss_list)
            where_max = np.where(loss_list == max_loss)
            print("train num: %-*d  step: %-*d  mean loss: %-*.7f  max loss: %-*.7f  where max: %-*d  replay buffer: %-*d"%(
                6, i, 6, step, 12, np.mean(loss_list), 12, max_loss, 6, where_max[0], 6, len(replay_buffer)))
    if not os.path.exists('save_predict/'):
        os.makedirs('save_predict/')
    savepath = "save_predict/{}_{}_{}_{}_50.pkl".format(para.env_name, para.total_step, para.num_test, nowtime)
    torch.save(critic_net.state_dict(), savepath)
            
        # print("final reward = ", np.mean(episode_rewards), "±", np.std(episode_rewards))
else:
    para.num_test = 10
    critic_net.load_state_dict(torch.load("save_predict/" + "humanoid_tqc_2000000.0_5000_0306_032220_50" + ".pkl"))
    fall_step_predict = deque()
    fall_step_total = deque()
    total_step = 0
    for i in range(para.num_test):
        state = env.reset()[0]
        step = 0
        while(True):
            total_step += 1
            step += 1
            action, _state = model.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            predict_step = critic_net(np.concatenate((state, action))).detach().numpy()[0]
            fall_step_predict.append(predict_step)
            # fall_step_total.append(predict_step)
            # print(fall_step_predict)
            if done:
                print(step)
                window_length = 200  # 窗口长度
                polyorder = 2      # 多项式阶数
                y_smoothed = savgol_filter(fall_step_predict, window_length, polyorder)
                plt.plot([x for x in range(step)], fall_step_predict, linewidth=0.6, label="origin")
                plt.plot([x for x in range(step)], y_smoothed, linewidth=0.8, label="smooth")
                plt.axhline(0, linewidth=0.5, color='black')
                plt.axhline(0.5, linewidth=0.5, color='black')
                plt.legend()
                plt.show()
                # os.system("pause")
                fall_step_total.extend(fall_step_predict)
                fall_step_predict.clear()
                break
        
    fig = plt.figure(figsize=(18, 5))
    plt.plot([y for y in range(int(total_step))], fall_step_total, linewidth=0.4)
    plt.show()
        