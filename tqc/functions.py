import torch

from tqc import DEVICE


def eval_policy(policy, eval_env, max_episode_steps, eval_episodes=10):
    policy.eval()
    avg_reward = 0.
    episode_reward_list = list()
    episode_reward = 0
    for _ in range(eval_episodes):
        episode_reward = 0
        state, done = eval_env.reset()[0], False
        t = 0
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            state, reward, done, _, info = eval_env.step(action)
            episode_reward += reward
            avg_reward += reward
            t += 1
        episode_reward_list.append(episode_reward)
    avg_reward /= eval_episodes
    # policy.train()
    return avg_reward, episode_reward_list


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
