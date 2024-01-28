from collections import deque

import dill
import matplotlib.pyplot as plt
import numpy as np
import platform
import os
from tracereader import TraceReader

def ChooseContinueTracePath():
    if platform.node() == 'robot-GALAX-B760-METALTOP-D4':
        path='/home/robot/Documents/ContinueTrace/'
    if platform.node() == 'DESKTOP-6S7M1IE':
        path='C:/ContinueTrace/'
    if platform.node() == 'ubuntu':
        path='/home/user/Desktop/robot/ContinueTrace/'
    else:
        path=''
    if not os.path.exists(path):
        os.makedirs(path)        
    return path
class Config():
    def __init__(self) -> None:
        self.trace_name = 'walker2d_td3_trace_continue_train_01-28_20-48-50'
        self.alpha_0 = -0.1
        self.alpha_1 = 0.01
        self.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "step_reward_target":deque(),
                      "reward_diff":deque(),
                      "alpha": deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp": deque(),
                      "mu_bias_centre":deque(),
                      "episode_step":deque(),
                      "critic_loss":deque()}

#* initialize
para = Config()
aTR = TraceReader(log_file_path=ChooseContinueTracePath() + para.trace_name + '.pkl')
data = aTR.get_trace()
print(len(data["episode_step"]))

#! Change start_episode & end episode to define which episode to be ploted, noted that episode index start with 0
start_episode = 1
end_episode = 200
plot_episode = [max(min(start_episode, len(data["episode_step"])-1), 0), min(len(data["episode_step"])-1, end_episode)]
start_step = 0
end_step = 0

if not plot_episode[0] == 0:
    for j in range(plot_episode[0]):
        start_step += data["episode_step"][j]
    end_step = start_step
if plot_episode[0] == plot_episode[-1]:
    if plot_episode[0] == len(data["episode_step"]) - 1 and plot_episode[0] == 0:
        plot_episode[-1] += 1
    elif plot_episode[0] == len(data["episode_step"]) - 1 and not plot_episode[0] == 0:
        plot_episode[0] -= 1
    else:
        plot_episode[-1] += 1

for j in range(plot_episode[0], plot_episode[-1]):
    end_step += data["episode_step"][j]

for key in para.Trace.keys():
    if not len(data[key]) == 0:
        data[key] = np.array(data[key])
        if key == 'episode_reward':
            data['episode_reward'] = data['episode_reward'][plot_episode[0]:plot_episode[-1]]
            continue
        if key == 'episode_reward_average':
            data['episode_reward_average'] = data['episode_reward_average'][plot_episode[0]:plot_episode[-1]]
            continue
        if key == 'episode_step':
            data['episode_step'] = data['episode_step'][plot_episode[0]:plot_episode[-1]]
            continue
        data[key] = data[key][start_step:end_step]
        


i = 3   # ℹ th dimension of action space, e.g i th neuro in last layer
ii_weight = 200    # weight of neuro ii_weight to neuro i, from previous layer to last layer(the action-generate layer)

fig, ax = plt.subplots(4, 1,figsize=(15,12), sharex=True)
fig.get_tight_layout()
fig.suptitle("from episode {} to episode {}".format(start_episode, end_episode))
# xticks = np.arange(start_step, end_step, 1000)

# ax[0].grid(True)
ax[0].plot(range(start_step, end_step), data["step_reward"], linewidth=0.6, label="step reward")
# ax[0].plot(range(start_step, end_step), data["step_reward_average"], linewidth=1.5, label="step reward exp average")
ax[0].plot(range(start_step, end_step), data["step_reward_target"], linewidth=0.6, label="step reward exp average(net learn)")
ax[0].plot(range(start_step, end_step), data["reward_diff"], linewidth=0.15, label="reward diff")
ax[0].axhline(0, linewidth=0.6, color='black')
# ax[0].plot(range(start_step, end_step), data["critic_loss"], linewidth=0.6, color='black', label="critic loss")
ax[0].legend()

# ax[1].grid(True)
ax[1].plot(range(start_step, end_step), data["alpha"], linewidth=1, label="alpha" )
# ax[1].plot(range(start_step, end_step), data["reward_diff"], linewidth=0.15, label="reward diff")
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

ax[2].plot(range(start_step, end_step), episode_reward, linewidth=0.6, label="episode reward")
ax[2].plot(range(start_step, end_step), episode_reward_average, linewidth=0.6, label="episode reward average")
ax[2].legend()

# ax[3].grid(True)
ax[3].plot(range(start_step, end_step), [data["mu_weight_centre"][j][i][ii_weight].tolist()  for j in range(len(data["step_reward"]))], 
           label='neuro{} to neuro{} weight centre'.format(ii_weight, i))
# ax[3].plot(range(start_step, end_step), [data["mu_weight"][j][i][ii_weight].tolist()  for j in range(len(data["step_reward"]))], 
#            linewidth=0.3,
#            label='neuro{} to neuro{} weight'.format(ii_weight, i))
ax[3].fill_between(range(start_step, end_step), 
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

# plt.xticks(xticks)

if not os.path.exists('continue_train_fig/'):
    os.makedirs('continue_train_fig/')
plt.savefig("continue_train_fig/" + para.trace_name + "_from_episode_{}_to_episode_{}".format(start_episode, plot_episode[-1]), dpi=600)
plt.show()


# def WherePlot(data, start_episode=None, end_episode=None):
#     start_step = None
#     end_step = None
#     start_e = None
#     end_e = None
#     total_step = len(data["step_reward"])
#     total_episode = len(data["episode_step"])
#     if_single_episode = False


#     if start_episode < 0 or end_episode < 0 or start_episode > end_episode:
#         raise ValueError("start_episode < 0 or end_episode < 0 or start_episode > end_episode")

#     # if single episode
#     if total_episode == 1:
#         return data, [0, total_step-1 +1], True
#     # default
#     if start_episode == None and end_episode == None:
#         return data, [0, total_step-1 +1], False
#     elif start_episode == None:
#         start_step = 0
#         start_e = 0
#         end_e = min(total_episode-1, end_episode)
#         if start_e > end_e:
#             raise ValueError("end episode too small")
#         elif start_e == end_e:
#             if_single_episode = True
#             end_step = data["episode_step"][0]  # single episode, e.g. first episode
#         elif start_e < end_e:
#             end_step = start_step
#             for i in range(end_e + 1):
#                 end_step += data["episode_step"][i]
    
#     elif end_episode == None:
#         end_step = total_step - 1
#         end_e = total_episode - 1
#         start_e = max(min(start_episode, total_episode-1), 0)
#         if start_e > end_e:
#             raise ValueError("start episode too large")
#         elif start_e == end_e:
#             if_single_episode = True
#             for i in range(end_e):
#                 start_step += data["episode_step"][i]   # single episode, e.g. last episode
#         elif start_e < end_e:
#             for i in range(end_e):
#                 start_step += data["episode_step"][i]
    
#     else:
#         start_e = max(min(start_episode, total_episode-1), 0)
#         end_e = min(total_episode-1, end_episode)
#         if start_e > end_e:
#             raise ValueError("start episode > end episode")
#         elif start_e == end_e and start_e == 0:
#             if_single_episode = True
#             start_step = 0
#             end_step = data["episode_step"][0]
#         elif start_e == end_e:
#             if_single_episode = True
#             start_step = 0
#             for i in range(start_e):
#                 start_step += data["episode_step"][i]
#             end_step = start_step + data["episode_step"][start_e]
#         elif start_e < end_e:
#             start_step = 0
#             for i in range(start_e):
#                 start_step += data["episode_step"][i]
#             end_step = start_step
#             for i in range(start_e, end_e+1):
#                 end_step += data["episode_step"][i]
    
#     if if_single_episode:
#         index = [start_e]
#     else:
#         index = np.arange(start_e, end_e+1, dtype=int)
    
#     end_step += 1   #end_step is the index in array,+1 to fit the range(start_step, end_step)

#     keys = data.keys()
#     for key in keys:
#         if not len(data[key]) == 0:
#             data[key] = np.array(data[key])
#             if key == 'episode_reward':
#                 data['episode_reward'] = data['episode_reward'][index]
#                 continue
#             if key == 'episode_reward_average':
#                 data['episode_reward_average'] = data['episode_reward_average'][index]
#                 continue
#             if key == 'episode_step':
#                 data['episode_step'] = data['episode_step'][index]
#                 continue
#             data[key] = data[key][start_step:end_step]

#     return data, [start_step, end_step], [start_e, end_e], if_single_episode

# data, [start_step, end_step], [start_episode, end_episode], single_episode = WherePlot(data, start_episode, end_episode)
# print(start_step, end_step)
# print(len(data["step_reward"]))