from algos.network.ac.ac import PPOAC as Policy
from distributed_marl.ppo.common.worker import RLWorker
import torch.nn as nn 
import torch 
from distributed_marl.algos.utils.ppo_shared_buffer import SharedDictReplayBuffer
import time 
import numpy as np 
from env_wrapper.popenv import POPEnv 
from satenv.base import BaseEnv
from satenv.configs.base_cfg import env_config
import random
from algos.network.ac.ac import PPOAC
import os 
from torch.utils.tensorboard import SummaryWriter
import ray 

@ray.remote
class PPOWorker(RLWorker):

    def __init__(self, 
        worker_id: int, port_cfg:dict, config:dict, common_cfg:dict):

        super().__init__(worker_id, port_cfg)
        self.config = config
        self.use_wandb = config["use_wandb"]
        self.device = config["worker_device"]
        # read necessary infomation
        # create env
        random.seed(self.worker_id)
        self.env = POPEnv(BaseEnv(env_config), common_cfg)
        self.seed = random.randint(1, 999)
        env_info = self.env.get_info()
        self.action_shape = env_info['action_shape']
        self.agent_num = env_info['pursuer_num']
        self.save_path = config['save_path']
        obs_space = self.env.observation_space
        act_space = self.env.action_space 
        self.policy = PPOAC(config, env_info, act_space, self.device)
        self.buffer = SharedDictReplayBuffer(config, 1, self.agent_num, 
                                            obs_space, act_space)

    def warmup(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        obs = self.env.reset()
        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()[np.newaxis]
    
    @torch.no_grad()
    def collect(self):
        obs = {}
        for key in self.buffer.obs.keys():
            obs[key] = np.concatenate(self.buffer.obs[key][-1])

        values, actions, action_log_probs = self.policy.get_actions(
                                            obs,    
                                            deterministic=False)
        
        return values.numpy(), actions.numpy(), action_log_probs.numpy()
    
    def collect_data(self):

         for step in range(self.buffer.episode_length):
            values, actions, action_log_probs = self.collect()
            obs, rewards, dones, infos = self.env.step(actions)
            if dones:
                obs = self.env.reset()
            dones = np.array([dones]*self.agent_num).reshape(-1, 1) # shape(agent_num, 1)
            rewards = rewards.reshape(-1,1) # shape(agent_num, 1)
            data = obs, rewards, dones, infos, values, actions, action_log_probs
            self.insert(data)

    def insert(self, data):

        obs, rewards, dones, infos, values, actions, action_log_probs = data 
        for key, value in obs.items():
            obs[key] = value[np.newaxis]
        self.buffer.insert(obs, 
                           actions[np.newaxis], 
                           action_log_probs[np.newaxis], 
                           values[np.newaxis], 
                           rewards[np.newaxis], 
                           dones[np.newaxis])

    def act(self):
        
        self.collect_data()
        local_buffer = self.buffer.get()
        self.send_replay_data(local_buffer)
        self.step += 1
        self.receive_new_params()
        self.buffer.after_update()
        

    def run(self):
        try:
            print(f'worker {self.worker_id} starts running')
            self.receive_new_params()
            print(f'work_{self.worker_id} has received params from learner')
            if self.worker_id == 1:
                """
                evaluate
                """
                self.test_run()
            else:
                self.step = 0
                self.warmup()
                while True:
                    self.act()
        except KeyboardInterrupt:
            import sys
            sys.exit()
    
    @torch.no_grad()
    def test_run(self):

        self.env.set_render_flag()
        self.epsilon = 0
        best_reward = 0
        episode_reward_sequence = []
        average_traking_ratio_sequence = []
        standard_deviation_sequence = []
        average_certainty_ratio_sequence = []
        average_collision_ratio_sequence = []
        time_start = time.time()
        episode = 0
        while True:
            if self.receive_new_params():
                episode_reward = 0
                obs = self.env.reset()
                done = False
                env_step = 0 
                while True:
                    action = self.policy.act(obs, deterministic=True)
                    next_obs, reward, done, info = self.env.step(action.numpy())
                    episode_reward += np.mean(reward)
                    obs = next_obs
                    env_step +=1
                    if done:
                        episode_reward_sequence.append(episode_reward)
                        average_traking_ratio_sequence.append(info.average_tracking_ratio)
                        standard_deviation_sequence.append(info.standard_deviation)
                        average_certainty_ratio_sequence.append(info.average_certainty_ratio)
                        average_collision_ratio_sequence.append(info.average_collision_ratio)
                        episode +=1
                        break
                mean_reward = np.mean(episode_reward_sequence[-50:])
                evaluate_dict = {'reward':episode_reward,
                                'average_traking_ratio':info.average_tracking_ratio,
                                'standard_deviation':info.standard_deviation,
                                'average_certainty_ratio':info.average_certainty_ratio,
                                'average_collision_ratio':info.average_collision_ratio,
                                 }
                self.send_evaluate_data(evaluate_dict)
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.env.save_info()
            else:
                pass        




         




    


