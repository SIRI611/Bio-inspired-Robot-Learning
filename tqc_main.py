import time
import dill
import gymnasium as gym
import numpy as np
import torch
import platform
import math
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
def f2(x):
    return 2 / (1 + np.exp(- x * 0.02)) - 1
class Config():
    def __init__(self) -> None:
        self.k1 = 0.00005
        self.k2 = 0.0002
        '''
        0.6 - 2
        0.9 - 10
        0.99 - 50
        0.995 - 200
        0.997 - 350
        0.998 - 500
        0.9875 - 800
        0.999 - 1000
        '''
        self.average_a = 0.997
        self.average_b = 0.999
        self.average_c = 0.9999
        self.average_d = 0.995

        self.alpha_0 = -0.1
        self.alpha_1 = 0.05
        self.a = 1e-5
        self.b = -2e-5
        self.period = 4000
        self.lr = 5e-5
        self.dt = 15
        self.total_step = 2e6
        self.is_train = True
        self.is_continue_train = True
        self.continue_train_episodes = int(1e4)
        self.if_trace = 2
        self.if_predict = 0

        self.env = 'Humanoid-v4'
        self.num_test = 20
        self.env_name="humanoid_tqc"
        self.if_critic = True
        #TODO change path
        self.logpath = "tensorboard/tqc_walker2d_tensorboard"
        self.gradient_path = "save_gradient/humanoid_tqc_abs_mean_gradient_600.pkl"
        self.weight_path = "save_weight/humanoid_tqc_weight.pkl"
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
                      "critic_loss":deque(),
                      "episode_step":deque()}
        
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

def calculate_amp_init(gradient_path, weight_path, k1, k2):
    with open(gradient_path, "rb") as f:
        gradient = dill.load(f)
        # gradient = torch.load(f, map_location=device)
    with open(weight_path, "rb") as f:
        weight = dill.load(f)
        # weight = torch.load(f, map_location=device)
    # gm = preprocessing(gradient)
    # print(gradient)
    gn = [1/(g.mean()+g) for g in gradient]
    wn = [abs(w) for w in weight]
    wn1 = [abs(w.mean()) for w in weight]
    amp_init = [abs(gn[i])*k1 + abs(wn[i])*k2 for i in range(len(gn))]
    # amp_init = [abs(gn[i]) * 1 for i in range(len(gn))]
    return amp_init, torch.abs(torch.mean(torch.tensor(wn1)))

para = Config()
episode_rewards = list()
# env = gym.make(para.env)
env = gym.make(para.env, render_mode="human")

if para.is_train:
    model = TQC("MlpPolicy", env, 
                total_step=para.total_step, 
                learning_starts=10000, 
                tensorboard_log=para.logpath, 
                env_name=para.env_name,
                verbose=1)

    # model = TQC('MultiInputPolicy', env,
    #             total_step=para.total_step, 
    #             buffer_size=1000000, 
    #             ent_coef='auto',
    #             batch_size=256,
    #             gamma=0.95,
    #             learning_rate=0.001,
    #             learning_starts=1000,
    #             # normalize=True,
    #             replay_buffer_class = HerReplayBuffer,
    #             replay_buffer_kwargs=dict(goal_selection_strategy='future',n_sampled_goal=4),
    #             policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
    #             tensorboard_log=para.logpath,
    #             verbose=1)
    
    model.learn(total_timesteps=para.total_step, log_interval=4)
    model.save("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    del model # remove to demonstrate saving and loading

else:
    model_0 = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    model_1 = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    # model_2 = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    # model = model_1

    replay_buffer_len = int(1e5)
    replay_buffer_list = [i for i in range(replay_buffer_len)]
    replay_buffer = deque(maxlen=replay_buffer_len)
    fall_step = deque(maxlen=replay_buffer_len)
    loss_list = list()
    num = int(1e5)
    action_list = deque(maxlen=num)
    state_list = deque(maxlen=num)
    state_action_list = deque(maxlen=num)

    reward_list = deque()
    reward_average = 0
    reward_diff = 0
    reward_diff_average = 0     #alpha
    alpha = 0
    alpha_average = 0
    if not para.is_continue_train:
        # model = TQC.load("save_model/" + 'walker2d_tqc_1000000.0' + '.pkl')
        # model = TQC.load("save_model_unfinished/"+ 'continue_train_walker2d_tqc_1000000.0_1000_450_0203_225534_unfinished' + '.pkl')
        
        for _ in range(para.num_test):    
            step = 0
            state = env.reset()[0]
            episode_reward = 0
            for _ in range(1000):
                
                action, _state = model_1.predict(state, deterministic=True)

                # env.render()         
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            print(episode_reward) 
            episode_rewards.append(episode_reward)    
            
        print("final reward = ", np.mean(episode_rewards), "±", np.std(episode_rewards))

    if para.is_continue_train:
        step_per_episode = int(1e4)
        checkpoint_step = 1000
        checkpoint = deque(maxlen=int(step_per_episode / checkpoint_step))
        model_0_flag = 0    #origin
        model_1_flag = 1    #continue train

        if para.if_critic:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            critic_net = Critic(state_dim=state_dim+action_dim, device=device, hidden_dim=[256, 128, 64, 16])
            critic_net.load_state_dict(torch.load("save_predict/" + "humanoid_tqc_2000000.0_5000_0306_032220_50" + ".pkl"))
        nowtime = time.strftime("%m%d_%H%M%S", time.localtime())
        amp_init, bound = calculate_amp_init(para.gradient_path, para.weight_path, para.k1, para.k2)
        model_1.actor.optimizer_dynamic = DynamicSynapse(model_1.actor.parameters(), lr=para.lr, amp=amp_init, period=para.period, dt=para.dt,
                                                       a=para.a,
                                                       b=para.b,
                                                       alpha_0=para.alpha_0,
                                                       alpha_1=para.alpha_1,
                                                       weight_oscillate_decay=1e-1
                                                        )
        #! define oscillate bound
        model_1.actor.optimizer_dynamic.oscillate_bound = bound / 5
        print(model_1.actor.optimizer_dynamic.oscillate_bound)
        
        for episode_idx in range(para.continue_train_episodes):
            if (episode_idx + 1) % 5 == 0:
                with open(ChooseContinueTracePath() + "{}_trace_continue_train_{}.pkl".format(para.env_name, nowtime), "ab") as f:
                    dill.dump(para.Trace, f, protocol=dill.HIGHEST_PROTOCOL)
                # reset Trace
                para.Trace = {"step_reward": deque(),
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
                      "mu_bias_amp":deque(),
                      "mu_bias_centre":deque(),
                      "episode_step":deque(),
                      "critic_loss":deque()}
                
            state = env.reset()[0]
            episode_reward = 0
            step = 0
            step_to_stable = 0
            model_0_step = 0
            model_1_step = 0
            for _ in range(step_per_episode): 
                step += 1
                if model_0_flag:
                    model_0_step += 1
                    step_to_stable += 1
                    action, _state = model_0.predict(state, deterministic=True)
                    state, reward, done, _, _ = env.step(action)
                    state_action_list.append(np.concatenate((state, action)))
                    fall_step_predict = critic_net(np.concatenate((state, action))).detach().numpy()[0]
                    model_1.actor.learn_dynamic(0, -1000)

                    if fall_step_predict < 0.5 or step_to_stable < 100:
                        model_1_flag = 0
                        model_0_flag = 1
                    else:
                        model_1_flag = 1
                        model_0_flag = 0
                        print("Switch 0 to 1 | Step:{}".format(model_0_step))
                        model_0_step = 0
                        step_to_stable = 0
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

                        if episode_idx > int(para.num_test / 100) or len(replay_buffer) == replay_buffer_len:
                            for _ in range(5):
                                if len(replay_buffer) < replay_buffer_len:
                                    sample = np.random.choice([i for i in range(len(replay_buffer))], size=512, replace=False)
                                else:
                                    sample = np.random.choice(replay_buffer_list, size=1024, replace=False)
                                for j in sample:
                                    loss = critic_net.learn(replay_buffer[j], f2(fall_step[j]))
                                    loss_list.append(loss)
                        break

                elif model_1_flag:
                    model_1_step += 1
                    action, _state = model_1.predict(state, deterministic=True)
                    state, reward, done, _, _ = env.step(action)
                    fall_step_predict = critic_net(np.concatenate((state, action))).detach().numpy()[0]
                    state_action_list.append(np.concatenate((state, action)))
                    if fall_step_predict < 0.5:
                        model_1_flag = 0
                        model_0_flag = 1
                        print("Switch 1 to 0 | Step:{}".format(model_1_step))
                        model_1_step = 0
                    else:
                        model_1_flag = 1
                        model_0_flag = 0

                # if step % checkpoint_step == 0:
                #     checkpoint.append(model)
                #     # print(len(checkpoint))
                    step += 1
                    if reward_average == 0:
                        reward_average = reward
                    else:
                        reward_average = para.average_a * reward_average + (1 - para.average_a) * reward

                    # TODO
                    # if para.if_critic:
                    #     reward_target = critic_net(state).detach()
                    #     loss = critic_net.learn(state, reward)
                    #     reward_diff = reward - reward_target
                    #     reward_diff = reward_diff.detach().numpy()[0]
                    # else:
                    reward_diff = reward - reward_average
                    
                    reward_diff_average = para.average_b * reward_diff_average + (1 - para.average_b) * reward_diff
                    alpha = reward_diff_average
                    # alpha_average = para.average_d * alpha_average + (1 - para.average_d) * alpha


                    model_1.actor.learn_dynamic(reward_diff, alpha)
                    episode_reward += reward


                    if para.if_trace and (step-1) % para.if_trace == 0:
                        para.Trace["step_reward"].append(reward)
                        para.Trace["step_reward_average"].append(reward_average)
                        # para.Trace["step_reward_target"].append(reward_target.detach().numpy())
                        para.Trace["alpha"].append(alpha)
                        para.Trace["alpha_average"].append(alpha_average)
                        para.Trace["reward_diff"].append(reward_diff)
                        para.Trace["mu_weight_amp"].append(model_1.actor.optimizer_dynamic.state_dict()["state"][4]["amp"].cpu().detach().numpy())
                        para.Trace["mu_weight_centre"].append(model_1.actor.optimizer_dynamic.state_dict()["state"][4]["weight_centre"].cpu().detach().numpy())
                        # para.Trace["mu_bias_amp"].append(model.actor.optimizer_dynamic.state_dict()["state"][5]["amp"].cpu().detach().numpy())
                        # para.Trace["mu_bias_centre"].append(model.actor.optimizer_dynamic.state_dict()["state"][5]["weight_centre"].cpu().detach().numpy())
                        # para.Trace["critic_loss"].append(loss)

                    # print(reward)
                    # if step == 10000:
                    #     plt.plot(range(10000), trace_reward, "g-")
                    #     plt.savefig("range_adapter.png")
                    #     plt.show()

                    #* step monitor
                    if step % 200 == 0:
                        print("mu weight amp:%.7f, mu weight centre:%.7f, reward exp average:%.7f, alpha:%.7f" 
                            %(model_1.actor.optimizer_dynamic.state_dict()["state"][4]["amp"][3][200],
                                model_1.actor.optimizer_dynamic.state_dict()["state"][4]["weight_centre"][3][200],
                                reward_average,
                                alpha))

                    if done:
                        # model = checkpoint[-1]
                        # print("Fall! Step:%d"%(step))
                        
                        break

            episode_rewards.append(episode_reward)
            para.Trace["episode_reward"].append(episode_reward)
            para.Trace["episode_reward_average"].append(sum(episode_rewards)/len(episode_rewards))
            para.Trace["episode_step"].append(math.ceil(step / para.if_trace))

            # model.actor.optimizer_dynamic.episode_reward_average = sum(episode_rewards)/len(episode_rewards)
            # model.actor.optimizer_dynamic.episode_step = step


            #* episode monitor
            if model_0_flag:
                print("=="*50)
                print("episode:", episode_idx, "\nepisode reward:", episode_reward, "episode len:",step, "episode reward average:", sum(episode_rewards)/len(episode_rewards))
            if model_1_flag:
                print("episode:", episode_idx, "\nepisode reward:", episode_reward, "episode len:",step, "episode reward average:", sum(episode_rewards)/len(episode_rewards))
                print("oscillate bound:", model_1.actor.optimizer_dynamic.oscillate_bound) 
                print("oscillate weight center:")
                print(model_1.actor.optimizer_dynamic.state_dict()['state'][4]['weight_centre'])
                # print("weight centre:%.7f" %(model.actor.optimizer_dynamic.state_dict()['state'][4]['weight_centre'][3][200]))
                print("oscillate amp:")
                print(model_1.actor.optimizer_dynamic.state_dict()['state'][4]['amp'])

            if (episode_idx + 1) % np.floor(para.continue_train_episodes / 20) == 0 and not (episode_idx + 1) == para.continue_train_episodes:
                path = 'save_model_unfinished/'
                if not os.path.exists(path):
                    os.makedirs(path)
                model_1.save(path + 'continue_train_{}_{}_{}_{}_{}_unfinished.pkl'.format(para.env_name, para.total_step, para.continue_train_episodes, episode_idx+1, nowtime))

        model_1.save("save_model/continue_train_{}_{}_{}_{}_finished.pkl".format(para.env_name, para.total_step, para.continue_train_episodes, nowtime))
