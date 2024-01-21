import gymnasium as gym
import dill
from tqc import TQC
import torch
import numpy as np

np.set_printoptions(threshold=np.inf)
# range_adapter = RangeAdapter()
# range_adapter.init_recording()
from dynamicsynapse import DynamicSynapse
from Adapter.RangeAdapter import RangeAdapter

def preprocessing(data):
    # torch.abs(data)
    for li in data:
        li = (li - torch.mean(li)) / torch.std(li)
    return data
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
class Config():
    def __init__(self) -> None:
        self.k1 = 0.1
        self.k2 = 0.002
        self.total_step = 2e6
        self.is_train = True
        self.is_continue_train = False
        self.continue_train_episodes = 3000
        # self.modelfilepath = "td3_humanoid.pkl"
        self.env = "Humanoid-v4"
        self.dt = 15
        self.num_test = 10
        self.env_name="humanoid_tqc"
        #TODO change path
        self.logpath = "tensorboard/tqc_humanoid_tensorboard"
        self.gradient_path = "save_gradient/ant_sac_max_gradient_600.pkl"
        self.weight_path = "save_weight/ant_sac_weight.pkl"

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
env = gym.make(para.env, render_mode="human")


if para.is_train:
    model = TQC("MlpPolicy", env, total_step=para.total_step, learning_starts=10000, tensorboard_log=para.logpath)
    
    model.learn(total_timesteps=para.total_step, log_interval=4)
    model.save("save_model/{}_{}.pkl".format(para.env_name, para.total_step))
    del model # remove to demonstrate saving and loading

else:
    model = TQC.load("save_model/{}_{}.pkl".format(para.env_name, para.total_step))

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

    if para.is_continue_train:
        amp_init = calculate_amp_init(para.gradient_path, para.weight_path, para.k1, para.k2)
        model.actor.optimizer_dynamic = DynamicSynapse(model.actor.parameters(), lr=model.actor.lr, amp=amp_init, period=4000, dt=para.dt)
        for episode_idx in range(para.continue_train_episodes):
            state = env.reset()[0]
            episode_reward = 0
            step = 0
            while True:  
                step += 1
                action, _state = model.predict(state, deterministic=True)
                # print(action)
                state, reward, done, _, _ = env.step(action)
                # print(done)

                # reward = range_adapter.step_dynamics(para.dt, reward)
                # trace_reward.append(reward)
                # range_adapter.recording()
                # range_adapter.update()

                model.actor.learn_dynamic(reward)
                episode_reward += reward
                
                # if step == 10000:
                #     plt.plot(range(10000), trace_reward, "g-")
                #     plt.savefig("range_adapter.png")
                #     plt.show()

                if done:
                    # print("break")
                    break 
            episode_rewards.append(episode_reward)
# with open("trace.pkl", "rb") as f:
#     trace = dill.load(f)
# print(trace["weight"])
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
