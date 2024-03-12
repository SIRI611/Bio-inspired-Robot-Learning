from collections import deque

import dill
import matplotlib.pyplot as plt
import numpy as np
import platform
import os
from tracereader import TraceReader


class Config():
    def __init__(self) -> None:
        self.trace_name = 'humanoid_tqc_trace_continue_train_0312_045618'
        self.alpha_0 = -0.1

        self.alpha_1 = 0.05
        #TODO Change start episode & end episode to define which episode to be ploted, noted that episode start from 0
        self.start_plot_episode = None
        self.end_plot_episode = None

        self.i = 3              # i th dimension of action space, e.g i th neuro in last layer
        self.ii_weight = 200    # weight of neuro ii_weight to neuro i, from previous layer to last layer(the action-generate layer)
        self.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "step_reward_target":deque(),
                      "reward_diff":deque(),
                      "alpha": deque(),
                      "alpha_average":deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp": deque(),
                      "mu_bias_centre":deque(),
                      "episode_step":deque(),
                      "critic_loss":deque()}

def ChooseContinueTracePath():
    if platform.node() == 'robot-GALAX-B760-METALTOP-D4':
        path='/home/robot/Documents/ContinueTrace/'
    elif platform.node() == 'DESKTOP-6S7M1IE':
        path='C:/ContinueTrace/'
    elif platform.node() == 'ubuntu':
        path='/home/user/Desktop/robot/ContinueTrace/'
    else:
        path='ContinueTrace/'
    if not os.path.exists(path):
        os.makedirs(path)        
    return path

def WherePlot(data, start_episode=None, end_episode=None):
    if start_episode != None and start_episode < 0:
        raise ValueError("start episode < 0")
    if end_episode != None and end_episode < 0:
        raise ValueError("end episode < 0")
    
    #! step, episode measure the episode_index of step and episode respectively, start from 0.
    start_step = 0
    end_step = 0
    start_e = None
    end_e = None
    is_single_episode = False
    total_episode = len(data["episode_reward"])
    total_step = len(data["step_reward"])
    episode_index = []

    if start_episode == None:
        start_episode = 0
        # start_step = 0
    if end_episode == None:
        end_episode = total_episode - 1
        # end_step = total_step - 1

    start_e = max(min(start_episode, total_episode - 1), 0)
    end_e = min(end_episode, total_episode - 1)

    if start_e > end_e:
        raise ValueError("start episode > end episode")
    elif start_e == end_e:
        is_single_episode = True
        if start_e == 0:
            start_step = 0
            end_step = data["episode_step"][0] - 1
        else:
            for i in range(start_e):
                start_step += data["episode_step"][i]
            end_step = start_step + data["episode_step"][end_e] - 1
    else:
        for i in range(start_e):
            start_step += data["episode_step"][i]
        end_step = start_step
        for i in range(start_e, end_e + 1):
            end_step += data["episode_step"][i]
        end_step = end_step - 1

    if is_single_episode == True:
        episode_index = [start_e]
    else:
        episode_index = [x for x in range(start_e, end_e + 1)]

    step_index = [x for x in range(start_step, end_step + 1)]
    # print(len(step_index))

    keys = data.keys()
    for key in keys:
        if not len(data[key]) == 0:
            data[key] = np.array(data[key])
            if key == 'episode_reward':
                data['episode_reward'] = data['episode_reward'][episode_index]
                continue
            if key == 'episode_reward_average':
                data['episode_reward_average'] = data['episode_reward_average'][episode_index]
                continue
            if key == 'episode_step':
                data['episode_step'] = data['episode_step'][episode_index]
                continue
            data[key] = data[key][step_index]
    return data, step_index, episode_index, is_single_episode


#* initialize
para = Config()
aTR = TraceReader(log_file_path=ChooseContinueTracePath() + para.trace_name + '.pkl')
data = aTR.get_trace()

data, step, episode, single = WherePlot(data, para.start_plot_episode, para.end_plot_episode)

i = para.i
ii_weight = para.ii_weight

fig, ax = plt.subplots(4, 1,figsize=(15,12), sharex=True)
fig.get_tight_layout()
if single:
    fig.suptitle("episode {}".format(episode[0]))
else:
    fig.suptitle("from episode {} to episode {}".format(episode[0], episode[-1]))

# ax[0].grid(True)
ax[0].plot(step, data["step_reward"], linewidth=0.6, label="step reward")
ax[0].plot(step, data["step_reward_average"], linewidth=1.5, label="step reward exp average")
# ax[0].plot(step, data["step_reward_target"], linewidth=0.6, label="step reward exp average(net learn)")
ax[0].plot(step, data["reward_diff"], linewidth=0.15, label="reward diff")
ax[0].axhline(0, linewidth=0.6, color='black')
# ax[0].plot(step, data["critic_loss"], linewidth=0.6, color='black', label="critic loss")
ax[0].legend()

# ax[1].grid(True)
ax[1].plot(step, data["alpha"], linewidth=1, label="alpha" )
# ax[1].plot(step, data["alpha_average"], linewidth=1, label="alpha average" )
# ax[1].plot(step, data["reward_diff"], linewidth=0.15, label="reward diff")
ax[1].axhline(para.alpha_0, linewidth=0.6, color='black')
ax[1].axhline(para.alpha_1, linewidth=0.6, color='black')
ax[1].legend()

# ax[2].grid(True)
episode_reward = []
episode_reward_average = []
for j in range(len(data["episode_reward"])):
    list1 = [data["episode_reward"][j] for x in range(data["episode_step"][j])]
    list2 = [data["episode_reward_average"][j] for x in range(data["episode_step"][j])]
    episode_reward.extend(list1)
    episode_reward_average.extend(list2)

ax[2].plot(step, episode_reward, linewidth=0.6, label="episode reward")
ax[2].plot(step, episode_reward_average, linewidth=0.6, label="episode reward average")
ax[2].legend()

# ax[3].grid(True)
ax[3].plot(step, [data["mu_weight_centre"][j][i][ii_weight].tolist()  for j in range(len(data["step_reward"]))], 
           label='neuro{} to neuro{} weight centre'.format(ii_weight, i))
# ax[3].plot(step, [data["mu_weight"][j][i][ii_weight].tolist()  for j in range(len(data["step_reward"]))], 
#            linewidth=0.3,
#            label='neuro{} to neuro{} weight'.format(ii_weight, i))
ax[3].fill_between(step, 
                   [data["mu_weight_centre"][j][i][ii_weight].tolist() + data["mu_weight_amp"][j][i][ii_weight].tolist() for j in range(len(data["step_reward"]))],
                   [data["mu_weight_centre"][j][i][ii_weight].tolist() - data["mu_weight_amp"][j][i][ii_weight].tolist() for j in range(len(data["step_reward"]))],
                   alpha=0.5)
ax[3].legend()


# ax[4].plot([data["mu_bias_centre"][j][i].tolist()  for j in range(len(data["step_reward"]))], 
#            label='neuro{} bias centre'.format(i))
# ax[4].fill_between([x for x in range(len(data["step_reward"]))],
#                    [data["mu_bias_centre"][j][i].tolist() + data["mu_bias_amp"][j][i].tolist() for j in range(len(data["step_reward"]))],
#                    [data["mu_bias_centre"][j][i].tolist() - data["mu_bias_amp"][j][i].tolist() for j in range(len(data["step_reward"]))],
#                    alpha=0.5)
# ax[4].legend()

xticks = np.arange(step[0], step[-1], 10000)
# plt.xticks(xticks)

if not os.path.exists('continue_train_fig/'):
    os.makedirs('continue_train_fig/')
if single:
    plt.savefig("continue_train_fig/" + para.trace_name + "_episode_{}".format(episode[0]), dpi=600)
else:
    plt.savefig("continue_train_fig/" + para.trace_name + "_from_episode_{}_to_episode_{}".format(episode[0], episode[-1]), dpi=600)
plt.show()
