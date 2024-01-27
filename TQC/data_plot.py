import dill
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tracereader import TraceReader


class Config():
    def __init__(self) -> None:
        self.trace_name = 'walker2d_tqc_trace_continue_train_01-27_16-45-43'
        self.alpha_0 = -0.05
        self.alpha_1 = 0.05
        self.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "step_reward_target":deque(),
                      "alpha": deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp": deque(),
                      "mu_bias_centre":deque(),
                      "critic_loss":deque()}

para = Config()

aTR = TraceReader(log_file_path='trace_continue_train/' + para.trace_name + '.pkl')
data = aTR.get_trace()

i = 3   # ℹ th dimension of action space, e.g i th neuro in last layer
ii_weight = 200    # weight of neuro ii_weight to neuro i, from previous layer to last layer(the action-generate layer)

fig, ax = plt.subplots(4, 1,figsize=(15,12), sharex=True)

# ax[0].grid(True)
ax[0].plot(data["step_reward"], linewidth=0.6, label="step reward")
# ax[0].plot(data["step_reward_average"], linewidth=1.5, label="step reward exp average")
ax[0].plot(data["step_reward_target"], linewidth=1.0, label="step reward exp average(net learn)")
# ax[0].plot(data["critic_loss"], linewidth=0.6, color='black', label="critic loss")
ax[0].legend()

# ax[1].grid(True)
ax[1].plot(data["alpha"], linewidth=1, label="alpha" )
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

ax[2].plot(episode_reward, linewidth=0.6, label="episode reward")
ax[2].plot(episode_reward_average, linewidth=0.6, label="episode reward average")
# ax[2].legend()

ax[3].plot([data["mu_weight_centre"][j][i][ii_weight].tolist()  for j in range(len(data["step_reward"]))], 
           label='neuro{} to neuro{} weight centre'.format(ii_weight, i))
ax[3].fill_between([x for x in range(len(data["step_reward"]))],
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

plt.show()
# for j in range(len(data["step_reward"])):
#     print(data["mu_bias_centre"][j][3].tolist())
# print(len((data["mu_bias_centre"])[0]))
# print((data["mu_bias_centre"])[0][0])