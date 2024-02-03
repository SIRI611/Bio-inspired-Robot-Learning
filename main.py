import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path
import dill
import time
from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction, Critic_Value
from tqc.dynamicsynapse import DynamicSynapse
from tqc.functions import eval_policy
from collections import deque


EPISODE_LENGTH = 1000
nowtime = time.strftime("%m-%d_%H-%M-%S", time.localtime())

def calculate_amp_init(gradient_path, weight_path, k1=0.001, k2=0.002):
    with open(gradient_path, "rb") as f:
        gradient = dill.load(f)
        # gradient = torch.load(f, map_location=device)
    with open(weight_path, "rb") as f:
        weight = dill.load(f)
        # weight = torch.load(f, map_location=device)
    # gm = preprocessing(gradient)
    # print(gradient)
    gn = [1/(g.mean()+g) for g in gradient]
    wn = [w for w in weight]
    amp_init = [gn[i]*k1 + wn[i]*k2 for i in range(len(gn))]
    return amp_init

def pretrain(args, results_dir, models_dir, trace_dir, prefix, nowtime):
    # --- Init ---

    # remove TimeLimit
    env = gym.make(args.env).unwrapped
    eval_env = gym.make(args.env).unwrapped

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_value = Critic_Value(state_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)
    critic_value_target = copy.deepcopy(critic_value)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
    trace_file_name = f"trace{prefix}_{args.env}_{args.seed}_{nowtime}.pkl"
    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      critic_value=critic_value,
                      critic_value_target=critic_value_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item(),
                      trace_path=trace_dir/trace_file_name)
    
    state, done = env.reset()[0], False
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    actor.train()
    for t in range(int(args.max_timesteps)):

        action = actor.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        episode_timesteps += 1

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_return += reward

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            trainer.train(replay_buffer, args.batch_size, args.max_timesteps)

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
            # Reset environment
            state, done = env.reset()[0], False

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

    file_name = f"{prefix}_{args.env}_{args.seed}"
    if args.save_model: 
        trainer.save(models_dir / file_name, nowtime)
        with open("models/replay_buffer.pkl", "wb") as f:
            dill.dump(replay_buffer, f)

def continue_train(args, models_dir, prefix, logtime, nowtime):
    Trace = {"step_reward": deque(),
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
    env = gym.make(args.env).unwrapped
    eval_env = gym.make(args.env).unwrapped

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    with open("models/replay_buffer.pkl", "rb") as f:
        replay_buffer = dill.load(f)
    
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_value = Critic_Value(state_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)
    critic_value_target = copy.deepcopy(critic_value)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      critic_value=critic_value,
                      critic_value_target=critic_value_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item(),
                      trace_path = None)
    
    file_name = f"{prefix}_{args.env}_{args.seed}"
    trainer.load(models_dir / file_name, logtime)
    amp_init = calculate_amp_init(gradient_path="save_gradient/humanoid_tqc_mean_gradient_600.pkl",
                                  weight_path="save_weight/humanoid_tqc_weight.pkl")
    trainer.dynamic_optimizer = DynamicSynapse(trainer.actor.parameters(), amp=amp_init)
    # evaluations = []
    state, done = env.reset()[0], False
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    actor.train()
    for t in range(int(args.continue_timesteps)):

        action = actor.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        episode_timesteps += 1

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_return += reward
        # Train agent after collecting sufficient data
        trainer.continue_train(replay_buffer, args.batch_size)

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
            # Reset environment
            state, done = env.reset()[0], False

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1
    
    if args.save_model: 
        trainer.save(models_dir / file_name, nowtime)

    #     # Evaluate episode
    #     if (t + 1) % args.eval_freq == 0:
    #         file_name = f"{prefix}_{args.env}_{args.seed}"+nowtime+".pkl"
    #         evaluations.append(eval_policy(actor, eval_env, EPISODE_LENGTH))
    #         np.save(results_dir / file_name, evaluations)
    #         if args.save_model: trainer.save(models_dir / file_name)
    # with open("models/replay_buffer.pkl", "wb") as f:
    #     dill.dump(replay_buffer, f)

def evaluate_policy(args, models_dir, prefix):
     
    eval_env = gym.make(args.env).unwrapped

    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(DEVICE)
    file_name = f"{prefix}_{args.env}_{args.seed}_actor"
    actor.load_state_dict((torch.load(models_dir / file_name)))

    avg_reward, episode_reward = eval_policy(actor, eval_env, EPISODE_LENGTH, eval_episodes=10)
    for index, reward in enumerate(episode_reward):
        print("episode{} reward: {}".format(index+1, reward))
    print("\nAverage reward: ", avg_reward)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v4")          # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=5e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e6, type=int)   # Max time steps to run environment
    parser.add_argument("--continue_timesteps", default=5e5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--log_dir", default='.')
    parser.add_argument("--prefix", default='')
    parser.add_argument("--save_model", default=True, action="store_true")
    parser.add_argument("--is_train", default=False)        # Save model and optimizer parameters
    parser.add_argument("--is_continue_train", default=False)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    results_dir = log_dir / 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dir = log_dir / 'models'
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    trace_dir = log_dir / 'trace'
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    continue_train_models_dir = log_dir / 'save_continue_train'
    if args.save_model and not os.path.exists(continue_train_models_dir):
        os.makedirs(continue_train_models_dir)

    if args.is_train == True:
        pretrain(args, results_dir, models_dir, trace_dir, args.prefix, nowtime)
    elif args.is_continue_train == True:
        logtime = ''
        continue_train(args, models_dir=continue_train_models_dir, prefix=args.prefix, logtime=logtime, nowtime=nowtime)
    else:
        evaluate_policy(args, models_dir, args.prefix)

