
"""
This buffer is used for on-policy algos like mappo
"""

import torch
import numpy as np
from utils.util import get_obs_shape, get_action_dim

class SharedDictReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """
    def __init__(self, args, n_rollout_threads, num_agents, obs_space, act_space):

        self.episode_length = args.episode_length
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_valuenorm = args.use_valuenorm
        self.n_rollout_threads = n_rollout_threads
        obs_shape = get_obs_shape(obs_space)

        self.available_actions = None 
        self.obs = {
            key: np.zeros((self.episode_length+1, self.n_rollout_threads, num_agents) + _obs_shape, dtype=np.float32)
            for key, _obs_shape in obs_shape.items()
                    }        

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        act_shape = get_action_dim(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.step = 0

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        """
        Insert data into the buffer.
        """
        for key in self.obs.keys():
            obs_ = (obs[key]).copy()
            self.obs[key][self.step + 1] = obs_

        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()

        self.rewards[self.step] = rewards.copy()

        self.masks[self.step+1] = masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        
        for key in self.obs.keys():
            obs_ = self.obs[key][-1].copy()
            self.obs[key][0] = obs_

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                if self._use_valuenorm:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                        self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                            self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        obs = {}
        for key in self.obs.keys():
            obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            obs_batch = {}
            for key, value in obs.items():
                obs_batch[key] = value[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield obs_batch, actions_batch,\
                  value_preds_batch, return_batch, old_action_log_probs_batch,\
                  adv_targ
    
    def get(self):
    
        """
        return all data in the buffer
        """
        obs = {}
        for key, value in self.obs.items():
            obs[key] = np.concatenate(value)    
        return dict(obs=obs,
                    masks=np.concatenate(self.masks),
                    value_preds=np.concatenate(self.value_preds),
                    actions=np.concatenate(self.actions),
                    action_log_probs=np.concatenate(self.action_log_probs),
                    rewards=np.concatenate(self.rewards),
                    )