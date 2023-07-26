import torch as T
import torch.nn.functional as F
from math import inf
import numpy as np
from numpy._typing import ArrayLike

from DDPG.networks import ActorNetwork, CriticNetwork
from DDPG.buffer import ReplayBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

def _state_to_network_input(state: tuple[ArrayLike, int, int]) -> np.ndarray:
    floats, price_counter, hour = state
    state_list = list(floats)
    state_list.extend([float(price_counter), float(hour)])

    # Normalize to mean = 0.0 and standard deviation = 1.0
    state_list[0] = (state_list[0] - 0.5) * 3.0
    state_list[1] = (state_list[1] - 0.5) * 3.0
    state_list[2] = (state_list[2] - 7.289) / 8.947
    state_list[3] = (state_list[3] - 498.91) / 385.17
    state_list[4] = (state_list[4] - 43.48) / 36.96
    state_list[5] = (state_list[5] - 0.5417) / 0.2971
    state_list[6] = state_list[6] / 2.0
    state_list[7] = (state_list[7] - 11.5) / 30.0

    return np.array(state_list, dtype=np.float64)

def _network_output_to_action(nn_output: ArrayLike) -> ArrayLike:
    max_idx = np.argmax(nn_output)
    tcl_action = max_idx // 20
    price_level = (max_idx - tcl_action * 20) // 4
    def_action = (max_idx - tcl_action * 20 - price_level * 4) // 2
    exc_action = (max_idx - tcl_action * 20 - price_level * 4) % 2
    action = np.array([tcl_action, price_level, def_action, exc_action], dtype=np.int64)
    return action



class DDPG:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim,
                 actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, action_noise=0.1, max_size=1000000,
                 batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                           fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic.parameters(),
                                                       self.target_critic.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(_state_to_network_input(state), action, reward, _state_to_network_input(state_), done)

    def choose_action(self, observation, train=True):
        observation = _state_to_network_input(observation)

        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(device)
        action = self.actor.forward(state).squeeze()

        if train:
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            action = T.clamp(action+noise, -1, 1)
        self.actor.train()

        return _network_output_to_action(action.detach().cpu().numpy())

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, reward, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_ = self.target_critic.forward(next_states_tensor, next_actions_tensor).view(-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_
        q = self.critic.forward(states_tensor, actions_tensor).view(-1)

        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -T.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic.save_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Saving critic network successfully!')
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Saving target critic network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic.load_checkpoint(self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir +
                                           'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Loading target critic network successfully!')
