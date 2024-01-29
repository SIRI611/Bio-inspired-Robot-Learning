import time
import dill
import gymnasium as gym
import numpy as np
import torch
import platform
from critic import Critic
from tqc import TQC
import os
# torch.set_printoptions(precision=16)
# torch.set_default_dtype(torch.float64)
import panda_gym
np.set_printoptions(threshold=np.inf)
# range_adapter = RangeAdapter()
# range_adapter.init_recording()
from dynamicsynapse import DynamicSynapse
from Adapter.RangeAdapter import RangeAdapter
from collections import deque
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
# def closure(r):
#     def a():
#         # out = adapter.step_dynamics(dt, r)
#         # adapter.update()
#         return r
#     return a

# def preprocessing(data):
#     # torch.abs(data)
#     for li in data:
#         li = (li - torch.mean(li)) / torch.std(li)
    # return data

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

class Config():
    def __init__(self) -> None:
        self.k1 = 0.005
        self.k2 = 0.002
        '''
        0.6 - 2
        0.9 - 10
        0.99 - 50
        0.995 - 200
        0.997 - 350
        0.998 - 500
        '''
        self.average_a = 0.995
        self.average_b = 0.995
        self.alpha_0 = -0.1
        self.alpha_1 = 0.01
        self.a = 2e-5
        self.b = -1e-5
        self.period = 1250
        self.lr = 1e-3
        self.dt = 8
        self.total_step = 3e6
        self.is_train = True
        self.is_continue_train = False
        self.continue_train_episodes = 1000

        self.env = 'PandaSlide-v3'
        self.num_test = 20
        self.env_name="pandaslide_tqc"
        #TODO change path
        self.logpath = "tensorboard/tqc_pandaslide_tensorboard"
        self.gradient_path = "save_gradient/humanoidstandup_tqc_max_gradient_600.pkl"
        self.weight_path = "save_weight/humanoidstandup_tqc_weight.pkl"
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
                      "critic_loss":deque(),
                      "episode_step":deque()}
        
def ChooseContinueTracePath():
    if platform.node() == 'robot-GALAX-B760-METALTOP-D4':
        path='/home/robot/Documents/ContinueTrace/'
    if platform.node() == 'DESKTOP-6S7M1IE':
        path='C:/ContinueTrace/'
    if platform.node() == 'ubuntu':
        path='/home/user/Desktop/robot/ContinueTrace/'
    else:
        path='ContinueTrace/'
    if not os.path.exists(path):
        os.makedirs(path)        
    return path

def calculate_amp_init(gradient_path, weight_path, k1, k2):
    with open(gradient_path, "rb") as f:
        gradient = dill.load(f)
        # gradient = torch.load(f, map_location=device)
    with open(weight_path, "rb") as f:
        weight = dill.load(f)
        # weight = torch.load(f, map_location=device)
    # gm = preprocessing(gradient)
    # print(gradient)
    gn = [torch.abs(g * k1) for g in gradient]
    wn = [torch.abs(w * k2) for w in weight]
    amp_init = [gn[i] + wn[i] for i in range(len(gn))]
    return amp_init

para = Config()
episode_rewards = list()
# env = gym.make(para.env, render_mode='human')
env = gym.make(para.env)

if para.is_train:
    # model = TQC("MultiInputPolicy", env, 
    #             total_step=para.total_step, 
    #             learning_starts=10000, 
    #             tensorboard_log=para.logpath, 
    #             env_name=para.env_name,
    #             verbose=1)

    model = TQC('MultiInputPolicy', env,
                total_step=para.total_step, 
                buffer_size=1000000, 
                ent_coef='auto',
                batch_size=256,
                gamma=0.95,
                learning_rate=0.001,
                learning_starts=1000,
                # normalize=True,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs=dict(goal_selection_strategy='future',n_sampled_goal=4),
                policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
                tensorboard_log=para.logpath)
    
    model.learn(total_timesteps=para.total_step, log_interval=4)
    model.save("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    del model # remove to demonstrate saving and loading

else:
    model = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    reward_list = deque()
    reward_average = 0
    reward_diff = 0
    reward_diff_average = 0     #alpha
    alpha = 0
    if not para.is_continue_train:
        # model = TQC.load("save_model/continue_train_{}_{}.pkl".format(para.env_name, para.continue_train_episodes))
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
        state_dim = env.observation_space.shape[0]
        critic_net = Critic(state_dim=state_dim, device=device, hidden_dim=[8, 4])
        critic_net.load_state_dict(torch.load("save_critic/{}_{}.pkl".format(para.env_name, para.total_step)))
        nowtime = time.strftime("%m-%d_%H-%M-%S", time.localtime())
        amp_init = calculate_amp_init(para.gradient_path, para.weight_path, para.k1, para.k2)
        model.actor.optimizer_dynamic = DynamicSynapse(model.actor.parameters(), lr=para.lr, amp=amp_init, period=para.period, dt=para.dt,
                                                       a=para.a,
                                                       b=para.b,
                                                       alpha_0=para.alpha_0,
                                                       alpha_1=para.alpha_1,
                                                       weight_oscillate_decay=1e-2)
        
        for episode_idx in range(para.continue_train_episodes):
            if episode_idx % 5 == 1:
                with open(ChooseContinueTracePath() + "{}_trace_continue_train_{}.pkl".format(para.env_name, nowtime), "ab") as f:
                    dill.dump(para.Trace, f, protocol=dill.HIGHEST_PROTOCOL)
                # reset Trace
                para.Trace = {"step_reward": deque(),
                      "step_reward_average": deque(),
                      "step_reward_target":deque(),
                      "reward_diff":deque(),
                      "alpha": deque(),
                      "episode_reward":deque(),
                      "episode_reward_average":deque(),
                      "mu_weight":deque(),
                      "mu_weight_amp": deque(),
                      "mu_weight_centre":deque(),
                      "mu_bias_amp":deque(),
                      "mu_bias_centre":deque(),
                      "episode_step":deque(),
                      "critic_loss":deque()}
                
            state = env.reset()[0]
            episode_reward = 0
            step = 0
            for _ in range(1000):  
                step += 1
                action, _state = model.predict(state, deterministic=True)
                # print(action)
                state, reward, done, _, _ = env.step(action)
                # print(done)
                
                if reward_average == 0:
                    reward_average = reward
                else:
                    reward_average = para.average_a * reward_average + (1 - para.average_a) * reward

                reward_target = critic_net(state).detach()
                loss = critic_net.learn(state, reward)
                reward_diff = reward - reward_target
                reward_diff = reward_diff.detach().numpy()[0]
                
                reward_diff_average = para.average_b * reward_diff_average + (1 - para.average_b) * reward_diff
                alpha = reward_diff_average
                # print(reward_average)
                # print(reward_diff_average)
                
                '''
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
                '''

                model.actor.learn_dynamic(reward_diff, alpha)
                episode_reward += reward

                para.Trace["step_reward"].append(reward)
                # para.Trace["step_reward_average"].append(reward_average)
                para.Trace["step_reward_target"].append(reward_target.detach().numpy())
                para.Trace["alpha"].append(alpha)
                para.Trace["reward_diff"].append(reward_diff)
                para.Trace["mu_weight_amp"].append(model.actor.optimizer_dynamic.state_dict()["state"][4]["amp"].cpu().detach().numpy())
                para.Trace["mu_weight_centre"].append(model.actor.optimizer_dynamic.state_dict()["state"][4]["weight_centre"].cpu().detach().numpy())
                # para.Trace["mu_bias_amp"].append(model.actor.optimizer_dynamic.state_dict()["state"][5]["amp"].cpu().detach().numpy())
                # para.Trace["mu_bias_centre"].append(model.actor.optimizer_dynamic.state_dict()["state"][5]["weight_centre"].cpu().detach().numpy())
                # para.Trace["critic_loss"].append(loss)

                # print(reward)
                # if step == 10000:
                #     plt.plot(range(10000), trace_reward, "g-")
                #     plt.savefig("range_adapter.png")
                #     plt.show()

                #* step monitor
                if step % 100 == 1:
                    print("mu weight amp:%.7f, mu weight centre:%.7f, reward exp average:%.7f, alpha:%.7f" 
                          %(model.actor.optimizer_dynamic.state_dict()["state"][4]["amp"][3][200],
                            model.actor.optimizer_dynamic.state_dict()["state"][4]["weight_centre"][3][200],
                            reward_average,
                            alpha))

                if done:
                    break

            episode_rewards.append(episode_reward)
            para.Trace["episode_reward"].append(episode_reward)
            para.Trace["episode_reward_average"].append(sum(episode_rewards)/len(episode_rewards))
            para.Trace["episode_step"].append(step)

            #* episode monitor
            print("episode:", episode_idx, "\nepisode_reward:", episode_reward)  
            print("oscillate weight center:")
            print(model.actor.optimizer_dynamic.state_dict()['state'][4]['weight_centre'])
            # print("weight centre:%.7f" %(model.actor.optimizer_dynamic.state_dict()['state'][4]['weight_centre'][3][200]))
            print("oscillate amp:")
            print(model.actor.optimizer_dynamic.state_dict()['state'][4]['amp'])

        model.save("save_model/continue_train_{}_{}.pkl".format(para.env_name, para.continue_train_episodes))
# with open("trace.pkl", "rb") as f:
#     trace = dill.load(f)
# print(trace["weight"])
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
