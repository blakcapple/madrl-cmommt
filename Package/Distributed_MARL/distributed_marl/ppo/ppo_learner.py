from copy import deepcopy
from distributed_rl.ppo.common.learner import Learner
from algos.policy.mappo import MAPPOTrainer
from algos.utils.ppo_shared_buffer import SharedDictReplayBuffer
import pyarrow as pa 
import zmq 
import numpy as np 
import torch 
from distributed_marl.ppo.utils import _t2n
import time 
import wandb 
import os 
import sys 
from algos.network.ac.ac import PPOAC
from satenv.wrapper.popenv import POPEnv 
from satenv.base import BaseEnv
from satenv.configs.base_cfg import env_config
from torch.utils.tensorboard import SummaryWriter
import ray  

@ray.remote(num_gpus=1)
class PPOLeaner(Learner):

    def __init__(self, port_config, config: dict, common_cfg:dict):

        super().__init__(port_config)
        self.config = config
        self.batch_size = config['episode_length']
        self.device = config["learner_device"]
        self.save_path = config['save_path']
        self.load_path = config['load_path']
        self.use_wandb = config['use_wandb']
        self.load = config['load']
        self.num_env_steps = config['num_env_steps']
        self.n_rollout_threads = config['num_workers'] - 1
        env = POPEnv(BaseEnv(env_config), common_cfg) 
        env_info = env.get_info()
        obs_space = env.observation_space
        action_space = env_info['action_space']
        self.agent_num = env_info['pursuer_num']
        self.policy = PPOAC(config, env_info, action_space, self.device)
        self.trainer = MAPPOTrainer(config, self.policy, self.device)
        self.buffer = SharedDictReplayBuffer(config, self.n_rollout_threads,
                                            self.agent_num, obs_space, 
                                            action_space)
        del env
        log_dir = os.path.join(self.save_path, 'learner')
        self.writter = SummaryWriter(log_dir)
        self.samples_num = 0
        if self.use_wandb:
            wandb.init(config=config,
                       project='SAT',
                       entity='the-one',
                       name=str('mappo'),
                       reinit=True)

    def recv_replay_data_(self):
        
        new_replay_data_id = self.pull_socket.recv()
        replay_data = pa.deserialize(new_replay_data_id)
        return replay_data

    def publish_params(self, new_params: np.ndarray):
        
        new_params_id = pa.serialize(new_params).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_evaluate_data(self):
        new_evaluate_data_id = False
        try: 
            new_evaluate_data_id = self.pair_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if new_evaluate_data_id:
            new_data = pa.deserialize(new_evaluate_data_id)
            self.evaluate_dict = new_data

    def get_params(self):
    
        actor_params = []
        critic_params = []
        actor = deepcopy(self.trainer.policy.actor)
        actor_state_dict = actor.cpu().state_dict()
        for param in list(actor_state_dict):
            actor_params.append(actor_state_dict[param].numpy())
        
        critic = deepcopy(self.trainer.policy.critic)
        critic_state_dict = critic.cpu().state_dict()
        for param in list(critic_state_dict):
            critic_params.append(critic_state_dict[param].numpy())

        return (actor_params, critic_params)

    def build_buffer(self):

        """
        从worker接收数据，拼接成合适的shape
        """

        data_key = ['obs', 'actions', 'rewards', 'masks', 'value_preds', 'action_log_probs']
        all_data = {key:[] for key in data_key}
        for _ in range(self.n_rollout_threads):
            replay_data = self.recv_replay_data_()
            

            for key in data_key:
                all_data[key].append(replay_data[key])
        for key in data_key:
            assert hasattr(self.buffer, key), 'check keys!'
            if key == 'obs':
                data_obs = all_data[key]
                for obs_key in data_obs[0].keys():
                    self.buffer.obs[obs_key] = np.stack([obs[obs_key] for obs in data_obs], axis=1)
            else: 
                setattr(self.buffer, key, np.stack(all_data[key], axis=1))

    @torch.no_grad()
    def compute(self):

        self.trainer.prep_rollout()
        last_obs = {}
        for key, value in self.buffer.obs.items():
            last_obs[key] = np.concatenate(value[-1])
        next_values = self.trainer.policy.get_values(last_obs)
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)

        return train_infos

    def run(self):

        try:
            start_time = time.time()
            self.best_reward = -np.inf
            self.update_step = 0
            self.evaluate_dict = {}
            params = self.get_params()
            self.publish_params(params)
            reward_sequence = []
            while self.samples_num <= self.num_env_steps:
                self.build_buffer()
                self.compute()
                train_infos = self.train()
                params = self.get_params()
                self.publish_params(params)
                self.recv_evaluate_data()
                self.update_step += 1
                self.samples_num += self.batch_size * self.update_step
                time_used = time.time() - start_time
                if self.evaluate_dict:
                    all_log_info = {**self.evaluate_dict, **train_infos}
                    reward_sequence.append(all_log_info['reward'])
                    mean_reward = np.mean(reward_sequence[-50:])
                    if mean_reward > self.best_reward:
                        self.best_reward = mean_reward
                        self.policy.save(self.save_path)
                        if self.config['use_valuenorm']:
                            self.trainer.value_normalizer.save(self.save_path)
                    self.log_info(all_log_info)
                    self.evaluate_dict = {}
                    print(f'FPS:{self.samples_num / time_used:.2f}, update_step:{self.update_step}, reward:{mean_reward}')

        except KeyboardInterrupt:
            sys.exit()
    
    def log_info(self, data:dict):
        if self.use_wandb:
            for key, value in data.items():
                wandb.log({key:value}, step=self.samples_num)
        else: 
            for key, value in data.items():
                self.writter.add_scalar(key, value, self.samples_num)

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.load_path) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.load_path)+'/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)
        if self.config['use_valuenorm']:
            self.trainer.value_normalizer.load(str(self.load_path)+'/value_norm.pt')





        