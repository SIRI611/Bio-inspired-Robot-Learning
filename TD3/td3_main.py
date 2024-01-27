import gymnasium as gym
import dill
from td3 import TD3
import torch
import numpy as np

np.set_printoptions(threshold=np.inf)
# range_adapter = RangeAdapter()
# range_adapter.init_recording()
from dynamicsynapse import DynamicSynapse
from Adapter.RangeAdapter import RangeAdapter
from collections import deque
def preprocessing(data):
    # torch.abs(data)
    for li in data:
        li = (li - torch.mean(li)) / torch.std(li)
    return data
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
class Config():
    def __init__(self) -> None:
        self.k1 = 0.001
        self.k2 = 0.00002
        self.total_step = 1e6
        self.is_train = False
        self.is_continue_train = False
        self.continue_train_episodes = 3000
        self.average_a = 0.9        # 10
        self.average_b = 0.98       # 50
        self.lr = 1e-3
        # self.modelfilepath = "td3_humanoid.pkl"
        self.env = "Walker2d-v4"
        self.dt = 15
        self.num_test = 10
        self.env_name="walker2d_td3"
        #TODO change path
        self.logpath = "tensorboard/td3_ant_tensorboard"
        self.gradient_path = "save_gradient/walker2d_td3_max_gradient_300.pkl"
        self.weight_path = "save_weight/walker2d_td3_weight.pkl"
        self.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "alpha": deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp": deque(),
                      "mu_bias_centre":deque()}

def calculate_amp_init(gradient_path, weight_path, k1, k2):
    with open(gradient_path, "rb") as f:
        gradient = dill.load(f)
    with open(weight_path, "rb") as f:
        weight = dill.load(f)
    # gm = preprocessing(gradient)
    # print(gradient)
    gn = [torch.abs(g * k1) for g in gradient]
    wn = [torch.abs(w * k2) for w in weight]
    amp_init = [gn[i] + wn[i] for i in range(len(gn))]
    return amp_init

para = Config()
episode_rewards = list()
# env = gym.make(para.env)
env = gym.make(para.env,render_mode="human")


if para.is_train:
    model = TD3("MlpPolicy", env,
                train_freq=1,gradient_steps=1,
                learning_rate=1e-3,batch_size=256,
                verbose=1,total_step=para.total_step, 
                env_name=para.env_name, tensorboard_log=para.logpath, 
                learning_starts=10000)
    
    model.learn(total_timesteps=para.total_step, log_interval=4)
    model.save("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    del model # remove to demonstrate saving and loading

else:
    model = TD3.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    reward_list = deque()
    reward_average = 0
    reward_diff = 0
    reward_diff_average = 0     #alpha
    if not para.is_continue_train:
        for _ in range(para.num_test):    
            state = env.reset()[0]
            episode_reward = 0
    
            for _ in range(1000):
                action, _state = model.predict(state, deterministic=True)

                # env.render()         
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            print(episode_reward) 
            episode_rewards.append(episode_reward)
        print("final reward = ", np.mean(episode_rewards), "±", np.std(episode_rewards))

    if para.is_continue_train:
        with open("trace_continue_train/{}_trace_continue_train.pkl".format(para.env_name), "wb") as f:
            dill.dump(para.Trace, f)
        amp_init = calculate_amp_init(para.gradient_path, para.weight_path, para.k1, para.k2)
        model.actor.optimizer_dynamic = DynamicSynapse(model.actor.parameters(), lr=para.lr, amp=amp_init, period=4000, dt=para.dt,
                                                       a=1e-5,
                                                       b=-1e-5,
                                                       alpha_0=-0.05,
                                                       alpha_1=0.05)
        for episode_idx in range(para.continue_train_episodes):
            if episode_idx % 5 == 1:
                with open("trace_continue_train/{}_trace_continue_train.pkl".format(para.env_name), "rb") as f:
                    data = dill.load(f)
                data["step_reward"] += para.Trace["step_reward"]
                data["step_reward_average"] += para.Trace["step_reward_average"]
                data["alpha"] += para.Trace["alpha"]
                data["episode_reward"] += para.Trace["episode_reward"]
                data["episode_reward_average"] += para.Trace["episode_reward_average"]
                data["mu_weight_amp"] += para.Trace["mu_weight_amp"]
                data["mu_weight_centre"] += para.Trace["mu_weight_centre"]
                data["mu_bias_amp"] += para.Trace["mu_bias_amp"]
                data["mu_bias_centre"] += para.Trace["mu_bias_centre"]
                with open("trace_continue_train/{}_trace_continue_train.pkl".format(para.env_name), "wb") as f:
                    dill.dump(data, f)
                para.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "alpha": deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp":deque(),
                      "mu_bias_centre":deque()}
                
            state = env.reset()[0]
            episode_reward = 0
            step = 0
            for _ in range(1000):  
                step += 1
                action, _state = model.predict(state, deterministic=True)
                # print(action)
                state, reward, done, _, _ = env.step(action)
                # print(done)

                para.Trace["step_reward"].append(reward)
                if reward_average == 0:
                    reward_average = reward
                else:
                    reward_average = para.average_a * reward_average + (1 - para.average_a) * reward
                    reward_diff = reward - reward_average
                para.Trace["step_reward_average"].append(reward_average)


                reward_diff_average = para.average_b * reward_diff_average + (1 - para.average_b) * reward_diff
                para.Trace["alpha"].append(reward_diff_average)
                # print(reward_average)
                # print(reward_diff_average)

                # if len(reward_list) > 0:
                #     sum_ = sum(reward_list)
                #     reward_ = reward - (sum_/len(reward_list))
                #     reward_list.append(reward)
                #     if len(reward_list) > 50:
                #         reward_list.popleft()
                        
                # if len(reward_list) == 0:
                #     reward_ = reward
                #     reward_list.append(reward)
                # print(reward_)

                # model.actor.learn_dynamic(reward_)
                model.actor.learn_dynamic(reward_diff_average)
                episode_reward += reward
                para.Trace["mu_weight_amp"].append(model.actor.optimizer_dynamic.state_dict()["state"][4]["amp"])
                para.Trace["mu_weight_centre"].append(model.actor.optimizer_dynamic.state_dict()["state"][4]["weight_centre"])
                para.Trace["mu_bias_amp"].append(model.actor.optimizer_dynamic.state_dict()["state"][5]["amp"])
                para.Trace["mu_bias_centre"].append(model.actor.optimizer_dynamic.state_dict()["state"][5]["weight_centre"])                
                # if step == 10000:
                #     plt.plot(range(10000), trace_reward, "g-")
                #     plt.savefig("range_adapter.png")
                #     plt.show()

                if done:
                    # print("break")
                    break
            para.Trace["episode_reward"].append(episode_reward)
            print("oscillate weight center:")
            print(model.actor.optimizer_dynamic.state_dict()['state'][5]['weight_centre'])
            # print("weight_oscillate_decay:")
            # print(model.actor.optimizer_dynamic.state_dict()['state'][5]['weight_oscilate_decay'])
            print("oscillate amp:")
            print(model.actor.optimizer_dynamic.state_dict()['state'][5]['amp'])
            print("episode:", episode_idx, "\nepisode_reward:", episode_reward, "\n")  
            episode_rewards.append(episode_reward)
            para.Trace["episode_reward_average"].append(sum(episode_rewards)/len(episode_rewards))
        model.save("save_model/continue_train_{}_{}.pkl".format(para.env_name, para.continue_train_episodes))
# with open("trace.pkl", "rb") as f:
# with open("trace.pkl", "rb") as f:
#     trace = dill.load(f)
# print(trace["weight"])
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()