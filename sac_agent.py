import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from replay_buffer import ReplayBuffer
from sac_actor import Actor
from sac_critic import Critic
import random
import itertools
import os
from torch.utils.tensorboard import SummaryWriter


BUFFER_SIZE = int(1e6)  # replay buffer size
ALPHA = 0.05            # initial temperature for SAC
TAU = 0.005             # soft update parameter
REWARD_SCALE = 1        # reward scale
NUM_LEARN = 1           # number of learning 
NUM_TIME_STEP = 1       # every NUM_TIME_STEP do update
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
RANDOM_STEP = 10000     # number of random step

# continuous action space
class Agent(nn.Module):
    def __init__(self, env, device, buffer_size=BUFFER_SIZE, reward_scale=REWARD_SCALE, batch_size=256, gamma=0.99, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, print_period=20, write_mode=True, save_period=1000000):
        super(Agent, self).__init__()
        
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.batch_size = batch_size
        self.print_period = print_period
        self.action_bound = [env.action_space.low[0], env.action_space.high[0]]
        self.tau = TAU
        self.reward_scale = reward_scale
        self.total_step = 0
        self.max_episode_time = env._max_episode_steps # maximum episode time for the given environment
        self.log_file = 'sac_log.txt'

        self.save_model_path = 'saved_models/'
        self.save_period = save_period
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        
        self.write_mode = write_mode
        if self.write_mode:
            self.writer = SummaryWriter('./log')

        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=0, device=self.device)
        
        self.actor = Actor(self.state_dim, self.action_dim, self.device, self.action_bound, lr=self.lr_actor).to(self.device)

        self.local_critic_1 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.local_critic_2 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        iterator = itertools.chain(self.local_critic_1.parameters(), self.local_critic_2.parameters())
        self.critic_optimizer = optim.Adam(iterator, lr=self.lr_critic)    


        self.H_bar = torch.tensor([-self.action_dim]).to(self.device).float() # minimum entropy
        self.alpha = ALPHA
        self.log_alpha = torch.tensor([1.0], requires_grad=True, device=self.device).float()
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_actor)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
    def my_print(self, content):
        with open(self.log_file, 'a') as writer:
            print(content)
            writer.write(content+'\n')

    def save_model(self):
        save_path = self.save_model_path + 'humanoid_v4_sac_{}.pth'.format(str(self.total_step))   
        torch.save(self.state_dict(), save_path)

    def load_model(self, path=None):
        if path is None:
            self.load_state_dict(torch.load(self.save_model_path + 'humanoid_v4_sac.pth'))
        else:
            self.load_state_dict(torch.load(path))
    def train(self, max_episode_num=1000, max_time=1000):
        
        self.my_print('######################### Start train #########################')
        self.episode_rewards = []
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.tot_steps = []

        # copy parameters to target 
        self.soft_update(self.local_critic_1, self.target_critic_1, 1.0)
        self.soft_update(self.local_critic_2, self.target_critic_2, 1.0)

        ts = []

        for episode_idx in range(max_episode_num):
            state = self.env.reset()[0]
            episode_reward = 0.
            temp_critic_loss_list = []
            temp_actor_loss_list = []
            for t in range(1, max_time+1):   
                if self.total_step < RANDOM_STEP:
                    action = self.env.action_space.sample()
                else:             
                    action = self.actor.get_action(torch.tensor([state]).to(self.device).float())
                    action = action.squeeze(dim=0).detach().cpu().numpy()
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                masked_done = False if t == self.max_episode_time else done

                self.memory.add(state, action, reward, next_state, masked_done)
                
                if len(self.memory) < self.batch_size or t % NUM_TIME_STEP != 0 : continue

                for _ in range(NUM_LEARN):
                    # Randomly sample a batch of trainsitions from D
                    states, actions, rewards, next_states, dones = self.memory.sample()

                    # Compute targets for the Q functions
                    # next_actions, next_log_probs = torch.tensor(self.actor.get_action_log_prob(next_states)).float().to(self.device)
                    # print(next_states)
                    with torch.no_grad():
                        sampled_next_actions, next_log_probs = self.actor.get_action_log_prob(next_states)
                        Q_target_1 = self.target_critic_1.forward(next_states, sampled_next_actions).detach()
                        Q_target_2 = self.target_critic_2.forward(next_states, sampled_next_actions).detach()
                        y = self.reward_scale*rewards + self.gamma * (1-dones)*(torch.min(Q_target_1, Q_target_2) - self.alpha*next_log_probs)

                    # Update Q-functions by one step of gradient descent
                    Q_1_current_value = self.local_critic_1.forward(states, actions)
                    Q_loss_1 = torch.mean((y - Q_1_current_value)**2)
                    Q_2_current_value = self.local_critic_2.forward(states, actions)
                    Q_loss_2 = torch.mean((y - Q_2_current_value)**2)
                    Q_loss = Q_loss_1 + Q_loss_2
                    self.critic_optimizer.zero_grad()
                    Q_loss.backward()
                    self.critic_optimizer.step()           
                    temp_critic_loss_list.append(Q_loss.item())

                    # Update policy by one step of gradient ascent
                    sampled_actions, log_probs = self.actor.get_action_log_prob(states)
                    Q_min = torch.min(self.local_critic_1.forward(states, sampled_actions), self.local_critic_2.forward(states, sampled_actions))
                    policy_loss = self.actor.learn(log_probs, Q_min, self.alpha)
                    temp_actor_loss_list.append(policy_loss)

                    # Adjust temperature
                    loss_log_alpha = self.log_alpha * (-log_probs.detach() - self.H_bar).mean()  
                    self.log_alpha_optimizer.zero_grad()
                    loss_log_alpha.backward()
                    self.log_alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().detach()

                    # Update target networks
                    self.soft_update(self.local_critic_1, self.target_critic_1, self.tau)
                    self.soft_update(self.local_critic_2, self.target_critic_2, self.tau)

                state = next_state

                self.total_step += 1

                if self.total_step % self.save_period == 0:
                    self.save_model()
                
                if done: break


            ts.append(t)
            self.tot_steps.append(self.total_step)
            self.episode_rewards.append(episode_reward)
            self.critic_loss_list.append(np.mean(temp_critic_loss_list))
            self.actor_loss_list.append(np.mean(temp_actor_loss_list))   
            if (episode_idx + 1) % self.print_period == 0:
                content = 'Tot_step: {0:<6} \t | Episode: {1:<4} \t | Time: {2:5.2f} \t | Reward : {3:5.3f} \t | actor_loss : {4:5.3f} \t | critic_loss : {5:5.3f}'.format(self.total_step, episode_idx + 1, np.mean(ts), np.mean(self.episode_rewards[-self.print_period:]), np.mean(self.actor_loss_list[-self.print_period:]), np.mean(self.critic_loss_list[-self.print_period:]))
                self.my_print(content)

                if self.write_mode:
                    self.writer.add_scalar('Loss/actor_loss', np.mean(self.actor_loss_list[-self.print_period:]), self.total_step)
                    self.writer.add_scalar('Loss/critic_loss', np.mean(self.critic_loss_list[-self.print_period:]), self.total_step)
                    self.writer.add_scalar('Reward/reward', np.mean(self.episode_rewards[-self.print_period:]), self.total_step)
                    self.writer.add_scalar('Reward/reward_per_time', np.mean(self.episode_rewards[-self.print_period:])/(np.mean(ts)+1e-6), self.total_step)
                    self.writer.add_scalar('Time/time',np.mean(ts), self.total_step)

                    alphas = self.log_alpha.exp().detach().cpu().numpy()
                    alpha_dict = {'task'+str(idx) : alpha_ for idx, alpha_ in enumerate(alphas)}
                    self.writer.add_scalars('Alpha/alpha', alpha_dict, self.total_step)

                ts = []