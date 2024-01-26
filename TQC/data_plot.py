import dill
import matplotlib.pyplot as plt

import time
from collections import deque


class Config():
    def __init__(self) -> None:
        self.trace_name = 'walker2d_tqc_trace_continue_train_01-25 21:14:05'
        self.alpha_0 = -0.05
        self.alpha_1 = 0.05
        self.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "alpha": deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp": deque(),
                      "mu_bias_centre":deque()}

para = Config()
with open("trace_continue_train/{}.pkl".format(para.trace_name), "rb") as f:
    data = dill.load(f)


i = 3   # ℹ th dimension of action space, e.g i th neuro in last layer
ii_weight = 200    # weight of neuro ii_weight to neuro i, from previous layer to last layer(the action-generate layer)

fig, ax = plt.subplots(5, 1,figsize=(15,12))

# ax[0].grid(True)
ax[0].plot(data["step_reward"], linewidth=0.6, label="step reward")
ax[0].plot(data["step_reward_average"], linewidth=1.5, label="step reward exp average")
ax[0].legend()

# ax[1].grid(True)
ax[1].plot(data["alpha"], linewidth=1, label="alpha" )
ax[1].axhline(para.alpha_0, linewidth=0.6, color='black')
ax[1].axhline(para.alpha_1, linewidth=0.6, color='black')
ax[1].legend()

# ax[2].grid(True)
ax[2].plot(data["episode_reward"], linewidth=0.6, label="episode reward")
ax[2].plot(data["episode_reward_average"], linewidth=0.6, label="episode reward average")
ax[2].legend()

# ax[3].plot([data["mu_weight_amp"][j][i][ii_weight].tolist() for j in range(len(data["step_reward"]))], 
#            label='neuro {} weight {} amp'.format(i, ii_weight))
ax[3].plot([data["mu_weight_centre"][j][i][ii_weight].tolist()  for j in range(len(data["step_reward"]))], 
           label='neuro{} to neuro{} weight centre'.format(ii_weight, i))
ax[3].fill_between([x for x in range(len(data["step_reward"]))],
                   [data["mu_weight_centre"][j][i][ii_weight].tolist() + data["mu_weight_amp"][j][i][ii_weight].tolist() for j in range(len(data["step_reward"]))],
                   [data["mu_weight_centre"][j][i][ii_weight].tolist() - data["mu_weight_amp"][j][i][ii_weight].tolist() for j in range(len(data["step_reward"]))],
                   alpha=0.5)
ax[3].legend()

ax[4].plot([data["mu_bias_centre"][j][i].tolist()  for j in range(len(data["step_reward"]))], 
           label='neuro{} bias centre'.format(i))
ax[4].fill_between([x for x in range(len(data["step_reward"]))],
                   [data["mu_bias_centre"][j][i].tolist() + data["mu_bias_amp"][j][i].tolist() for j in range(len(data["step_reward"]))],
                   [data["mu_bias_centre"][j][i].tolist() - data["mu_bias_amp"][j][i].tolist() for j in range(len(data["step_reward"]))],
                   alpha=0.5)
ax[4].legend()

plt.show()
# for j in range(len(data["step_reward"])):
#     print(data["mu_bias_centre"][j][3].tolist())
# print(len((data["mu_bias_centre"])[0]))
# print((data["mu_bias_centre"])[0][0])