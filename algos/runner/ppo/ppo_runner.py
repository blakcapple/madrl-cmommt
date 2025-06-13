import time
import wandb
import numpy as np
from functools import reduce
import torch
from .base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class PPORunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config, train_env, policy, eval_env):
        super(PPORunner, self).__init__(config, train_env, policy, eval_env)
        self.env_step = 0

    def run(self):
        obs = self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        best_reward = -np.inf
        for episode in range(episodes):

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, = self.collect(obs)
                self.env_step += self.n_rollout_threads
                # Obser reward and next obs
                obs, rewards, dones, infos,  reward_info, =  self.envs.step(actions)

                data = obs, rewards, dones,  \
                       values, actions, action_log_probs,
                # insert data into buffer
                self.insert(data)
            # compute return and update network
            self.compute()
            train_infos = self.train()
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                self.logger.info("updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                self.log_train(train_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                eval_reward = self.eval(self.eval_episode, total_num_steps)
                # save model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.save(episode, save_best=True)
        self.save(episodes)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        for key in self.buffer.obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()
        return obs

    def wrap_obs(self, obs:dict):
        wrapped_obs = {}
        for key, val in obs.items():
            wrapped_obs[key] = torch.from_numpy(np.concatenate(val)).to(dtype=torch.float32,device=self.device)
        return wrapped_obs

    def to_tensor(self, obs:dict):
        wrapped_obs = {}
        for key, val in obs.items():
            wrapped_obs[key] = torch.from_numpy(val).to(dtype=torch.float32,device=self.device)
        return wrapped_obs
    
    @torch.no_grad()
    def collect(self, obs):
        self.trainer.prep_rollout()
        value, action, action_log_prob,   \
            = self.trainer.policy.get_actions(self.wrap_obs(obs), env_step=self.env_step)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))

        return values, actions, action_log_probs

    def insert(self, data):
        
        obs, rewards, dones, values, actions, action_log_probs,  = data
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, action_log_probs, values, rewards, masks)

    def log_train(self, train_infos, total_num_steps):
        
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalar(k, v, total_num_steps)
                
    @torch.no_grad()
    def eval(self, total_episode, total_num_steps):
        episode_reward_sequence = [[] for _ in range(self.n_eval_rollout_threads)]
        episode_sum_reward = []
        average_traking_ratio_sequence = []
        standard_deviation_sequence = []
        average_certainty_ratio_sequence = []
        average_collision_ratio_sequence = []
        obs = self.eval_envs.reset()
        self.trainer.prep_rollout()
        episode_num = 0
        step = 0
        while episode_num < total_episode:
            actions = self.trainer.policy.act(self.wrap_obs(obs),
                                                deterministic=True)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            obs,  rewards, dones, infos, reward_info = self.eval_envs.step(actions)
            step += 1
            for idx, r in enumerate(rewards):
                episode_reward_sequence[idx].append(np.mean(r))
            if any(dones):
                episode_num += self.n_eval_rollout_threads
                episode_sum_reward.append(np.sum(episode_reward_sequence))
                average_traking_ratio_sequence.append(infos['average_tracking_ratio'])
                standard_deviation_sequence.append(infos['standard_deviation'])
                average_certainty_ratio_sequence.append(infos['average_certainty_ratio'])
                average_collision_ratio_sequence.append(infos['average_collision_ratio'])
        evaluate_dict = {
                        'average_traking_ratio':np.mean(average_traking_ratio_sequence),
                        'standard_deviation':np.mean(standard_deviation_sequence),
                        'average_certainty_ratio':np.mean(average_certainty_ratio_sequence),
                        'average_collision_ratio':np.mean(average_collision_ratio_sequence),
                            }
        info = f'eval_step: {total_num_steps} '
        for key, val in evaluate_dict.items():
            info += f'{key}: {round(val,2)} '
        self.log_train(evaluate_dict, total_num_steps)
        self.logger.info(info)
        return np.mean(episode_sum_reward)

            