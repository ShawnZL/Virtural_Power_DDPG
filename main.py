import gym
import numpy as np
import argparse
import custom_envs.grid_v0
from numpy._typing import ArrayLike

from DDPG.DDPG import DDPG
from DDPG.utils import create_directory, plot_learning_curve, scale_action

from TD3.TD3 import TD3
from TD3.utils import create_directory, plot_learning_curve, scale_action

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images/reward.png')

args = parser.parse_args()

"""
 _state_to_network_input 接受一个元组 state 作为输入，并返回一个包含浮点数的列表 state_list。
"""

def main_DDPG():
    num_days = 365
    env = gym.make("Grid-v0", max_total_steps = 24 * num_days)
    agent = DDPG(alpha=0.0003, beta=0.0003, state_dim=8,
                 action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                 critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.checkpoint_dir,
                 batch_size=256)
    create_directory(args.checkpoint_dir,
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        done = False
        total_reward = 0
        ep_reward = 0
        observation, _info = env.reset()
        while not done:

            action = agent.choose_action(observation, train=True)
            observation_, reward, done, _, _info = env.step(action)
            # print(observation_, reward, done, _info)
            # 这个函数scale_action用于将动作（action）从区间[-1, 1]映射到指定的范围[low, high]。
            # action_ = scale_action(action.copy(), env.action_space.high, env.action_space.low)

            env.render()
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            ep_reward += reward
            observation = observation_

        total_reward += ep_reward / num_days
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {:.8f} AvgReward: {:.8f}'.format(episode+1, total_reward, avg_reward))

        if (episode + 1) % 200 == 0:
            agent.save_models(episode+1)

    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
                        ylabel='reward', figure_file=args.figure_file)


def main_TD3():
    num_days = 365
    env = gym.make("Grid-v0", max_total_steps = 24 * num_days)
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=8,
                action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.checkpoint_dir, gamma=0.99,
                tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=1000000, batch_size=256)
    create_directory(path=args.checkpoint_dir, sub_path_list=['Actor', 'Critic1', 'Critic2', 'Target_actor',
                                                        'Target_critic1', 'Target_critic2'])

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        done = False
        total_reward = 0
        observation, _info = env.reset()
        while not done:

            action = agent.choose_action(observation, train=True)
            observation_, reward, done, _, _info = env.step(action)
            # print(observation_, reward, done, _info)
            # 这个函数scale_action用于将动作（action）从区间[-1, 1]映射到指定的范围[low, high]。
            # action_ = scale_action(action.copy(), env.action_space.high, env.action_space.low)

            env.render()
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {:.8f} AvgReward: {:.8f}'.format(episode+1, total_reward, avg_reward))

        if (episode + 1) % 200 == 0:
            agent.save_models(episode+1)

    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
                        ylabel='reward', figure_file=args.figure_file)



if __name__ == '__main__':
    # main_DDPG()
    main_TD3()