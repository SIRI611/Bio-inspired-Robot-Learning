import time

import dill
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import os
import copy
import itertools

from predict import Predict
from tqc import TQC
from collections import deque
from scipy.signal import savgol_filter
import dill
# torch.set_default_dtype(torch.float64)

class Config():
    env = "Humanoid-v4"
    env_name = "humanoid_tqc"
    total_step=2e6
    num_test = 1000
    if_train = True
    step_num = 20
    batch_size = 1024
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def f2(x):
    return 2 / (1 + np.exp(- x * 0.02)) - 1

para = Config()
model = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
env = gym.make(para.env)
# env = gym.make(para.env, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
fall_step_offset = (1 + para.step_num) / 2

predict_net = Predict(batch_size=para.batch_size, device=para.device)
predict_net.to(para.device)



if para.if_train:
    num = int(5e5)
    replay_buffer_len = int(2e2)

    replay_buffer = deque(maxlen=replay_buffer_len)
    # episode_record = deque(maxlen=num)
    # fall_step = deque(maxlen=replay_buffer_len)
    loss_list = deque()
    fall_step_list = deque(maxlen=para.batch_size)
    inputs = deque(maxlen=para.batch_size)

    nowtime = time.strftime("%m%d_%H%M%S", time.localtime())
    for episode_idx in range(para.num_test):    
        state = env.reset()[0]
        episode_reward = 0
        episode_record = []
        loss_list.clear()

        step = 0
        while(True):
            step += 1 
            action, _state = model.predict(state, deterministic=True)
            episode_record.append(np.concatenate((state, action, np.random.randint(2, size=1))))
            state, reward, done, _, _ = env.step(action)
            
            if done:
                replay_buffer.append(episode_record)
                
                # fall_step.append([i for i in range(step-1, -1, -1)])
                
                if episode_idx >= para.num_test / 50 or len(replay_buffer) == replay_buffer_len:
                # if episode_idx >= 1 or len(replay_buffer) == replay_buffer_len:
                    for _ in range(5):                      
                        sample = np.random.choice([i for i in range(len(replay_buffer))], size=para.batch_size, replace=True)
                        fall_step_list.clear()
                        inputs.clear()
                        for s in sample:
                            episode_length = len(replay_buffer[s])
                            # print(episode_length)
                            extension_length = (episode_length - para.step_num) * 2
                            
                            index = np.random.choice(extension_length, size=1, replace=True)[0]
                            #? average the probability
                            if index >= episode_length - para.step_num:
                                index = episode_length - para.step_num
                            fall_step_list.append(episode_length - index - para.step_num/2)
                            inputs.append(list(itertools.islice(replay_buffer[s], int(index), int(index+20))))
                            
                            
                        # print(replay_buffer[sample].shape)
                        inputs_ = np.expand_dims(np.array(inputs), axis=1)
                        target = np.array([f2(fall_step_num) for fall_step_num in fall_step_list]).reshape((para.batch_size, -1))
                        loss = predict_net.learn(torch.tensor(inputs_, dtype=torch.float32).to(para.device), torch.tensor(target, dtype=torch.float32).to(para.device))
                        loss_list.append(loss)
                break
        if len(loss_list) > 0:
            max_loss = np.max(loss_list)
            print("train num: %-*d  step: %-*d  mean loss: %-*.7f  max loss: %-*.7f  replay buffer: %-*d"%(
                6, episode_idx, 6, step, 12, np.mean(loss_list), 12, max_loss, 6, len(replay_buffer)))
    if not os.path.exists('save_predict/'):
        os.makedirs('save_predict/')
    savepath = "save_predict/{}_{}_{}_{}_20.pkl".format(para.env_name, para.total_step, para.num_test, nowtime)
    torch.save(predict_net.state_dict(), savepath)
    with open("replay_buffer/" + "replay_buffer_{}_{}.pkl".format(replay_buffer_len, nowtime), "ab") as f:
        dill.dump(replay_buffer, f, protocol=dill.HIGHEST_PROTOCOL)
            
        # print("final reward = ", np.mean(episode_rewards), "±", np.std(episode_rewards))
else:
    num = int(1e5)
    para.num_test = 10
    predict_net.load_state_dict(torch.load("save_predict/" + "humanoid_tqc_2000000.0_1000_0403_104933_20" + ".pkl"))

    # fall_step_predict = deque()
    # fall_step_total = deque()
    buffer = deque()
    # predict_step = deque(maxlen=num)
    # cluster = deque(maxlen=para.step_num)
    total_step = 0
    for i in range(para.num_test):
        episode_record = []
        state = env.reset()[0]
        buffer.clear()
        step = 0
        while(True):
            total_step += 1
            step += 1
            action, _state = model.predict(state, deterministic=True)
            episode_record.append(np.concatenate((state, action, np.array([0]))))
            state, reward, done, _, _ = env.step(action)
            

            if done:
                # print(step)
                # print(np.expand_dims(np.array(state_action_actor), axis=1).shape)
                for i in range(len(episode_record) - int(para.step_num)):
                    buffer.append(episode_record[-i-1:-i-1-para.step_num:-1])
                
                buffer_ = np.expand_dims(np.array(buffer), axis=1)
                print(buffer_.shape)
                predict_step = predict_net(buffer_).detach().numpy()[::-1]

                window_length = min(200, step)  # 窗口长度
                polyorder = 2      # 多项式阶数
                y_smoothed = savgol_filter(np.ravel(predict_step), window_length, polyorder)
                plt.plot([x for x in range((buffer_.shape[0]))], predict_step, linewidth=0.6, label="origin")
                plt.plot([x for x in range(buffer_.shape[0])], y_smoothed, linewidth=0.8, label="smooth")
                plt.axhline(0, linewidth=0.5, color='black')
                plt.axhline(0.5, linewidth=0.5, color='black')
                plt.legend()    
                plt.show()
                # os.system("pause")
                # fall_step_total.extend(fall_step_predict)
                # fall_step_predict.clear()
                break
        
    # fig = plt.figure(figsize=(18, 5))
    # plt.plot([y for y in range(int(total_step))], fall_step_total, linewidth=0.4)
    # plt.show()
        
