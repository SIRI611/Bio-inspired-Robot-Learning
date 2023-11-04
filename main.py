from sac_agent import Agent
import torch
import gym
import numpy as np
import matplotlib as plt

max_episode_num = 3000
env = gym.make('Humanoid-v4', render_mode="human")
print(env.action_space)
GPU_NUM = 0
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

is_train = False
is_cotinue_train = True
continue_train_episodes = 1000

# is_train = True

if is_train is True:
    agent = Agent(env, device)
    agent.train(max_episode_num=max_episode_num, max_time=1000)
else:
    agent = Agent(env, device, write_mode=False)
    agent.load_model(agent.save_model_path + 'humanoid_v4_sac_1000000.pth')

    # for layer,param in agent.state_dict().items(): # param is weight or bias(Tensor) 
    #     print(layer,param)
    avg_episode_reward = 0.
    num_test = 5
    episode_rewards = []
    if is_cotinue_train is False:
        for _ in range(num_test):    
            state = env.reset()[0]
            episode_reward = 0.
    
            for _ in range(1000):
                action = agent.actor.get_action(torch.tensor([state]).to(device).float(), stochastic=False)

                # env.render()         
                state, reward, done, _, _ = env.step(action[0])
                episode_reward += reward
                if done:
                    break 
            episode_rewards.append(episode_reward)

        print('avg_episode_reward : ', np.mean(episode_rewards))

    elif is_cotinue_train is True:

        for episode_idx in range(continue_train_episodes):
            state = env.reset()[0]
            episode_reward = 0
            while True:
                action = agent.actor.get_action(torch.tensor([state]).to(device).float(), stochastic=False)
                state, reward, done, _, _ = env.step(action[0])
                # print(done)
                agent.actor.learn_dynamic(reward)

                episode_reward += reward
                agent.total_step += 1
                if done:
                    print("break")
                    break 
            episode_rewards.append(episode_reward)
            if (episode_idx + 1) % agent.print_period == 0:
                    content = 'Tot_step: {} \t | Episode: {} \t  | Reward : {} \t '.format(agent.total_step, episode_idx + 1, np.mean(episode_rewards[-agent.print_period:]))
                    agent.my_print(content)

                # if agent.write_mode:
                #     agent.writer.add_scalar('Reward/reward', np.mean(episode_rewards[-agent.print_period:]), agent.total_step)

            
        

        