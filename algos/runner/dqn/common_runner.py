"""
written by Ruan Yudi 
2021.8.2
"""
import time
import numpy as np
import wandb
from utils.plot import plot_qloss, plot_rw
from algos.buffer.dqn_replay_buffer import MultiPrioritizedReplayBuffer, MultiAgentReplayBuffer
import os 
from torch.utils.tensorboard import SummaryWriter
from algos.agents.dqn_agent import DQNAgent
import torch 

def _t2n(x):
    return x.detach().cpu().numpy()

class Trainer(object):
    '''Train process'''
    def __init__(self, env, env_info, agent: DQNAgent, save_file, logger, log_file, train_config):

        self.use_wandb = train_config['use_wandb']
        self.total_train_step = train_config['train_step']
        self.total_warmup_step = train_config['warmup_step']
        self.log_path = train_config['save_path']
        self.use_eval =  train_config['use_eval']
        self.eval_interval = train_config['eval_interval']
        self.eval_episode = train_config['eval_episode']
        self.log_interval = train_config['log_interval']
        self.update_inteval = train_config['update_interval']
        self.n_eval_rollout_threads = train_config['n_eval_rollout_threads']
        self.device = train_config['device']
        self.env = env
        self.env_info = env_info
        self.agent = agent
        self.save_file = save_file
        self.logger = logger
        self.log_file = log_file
        self.cfg = train_config
        self.total_update_step = int(self.total_train_step // self.update_inteval)
        if not self.use_wandb:
            log_dir = os.path.join(self.log_path, 'buffer')
            self.writer = SummaryWriter(log_dir)
        self.build_memory()
        self.update_step = 0
        self.best_test_score = -np.inf
        self.best_train_score = -np.inf
        self.eval_envs = None

    def update(self):
        if self.memory.ready():
            if self.per:
                self.beta = self.beta_min + (1-self.beta_min)*(self.update_step/(self.total_update_step))
                data = self.memory.sample_batch(self.beta)
            else:
                data = self.memory.sample_batch()
            if self.per:
                priority, idxs = self.agent.train(data, self.update_step)
            else:
                self.agent.train(data, self.update_step)
            if self.per:
                self.memory.update_priorities(idxs, list(priority))
            self.update_step += 1 

    def train(self):
        """
        智能体训练回合
        """
        episode_score = []
        time_start = time.time()
        env_step = 0
        episode = 0
        while env_step < self.total_train_step:
            episode_score.append(0)
            episode_reward_info = {}
            done = False
            obs = self.env.reset()
            episode_q = []
            episode_step = 0 
            while not done:
                "train loop begin"
                actions, q_value = self.agent.select_actions(obs, False, env_step)
                obs_new, reward, done, info, reward_info = self.env.step(actions)
                for key, val in reward_info.items():
                    episode_reward_info[key] = episode_reward_info.get(key, 0) + np.mean(val)
                env_step +=1
                episode_step += 1
                self.memory.push(obs, actions, reward, obs_new, done)
                if env_step % self.update_inteval == 0:
                    self.update()
                obs = obs_new
                episode_score[-1] += np.mean(reward).item()
                episode_q.append(q_value)
                "tran loop end"

            if (episode+1) % 100 == 0 :
                plot_rw(self.log_file, window_size=100, phase='train')
                plot_qloss(self.log_file, window_size=100)
                
            mean_score = np.mean(episode_score[-50:])
            if self.best_train_score < mean_score:
                self.best_train_score = mean_score
                self.agent.save_weight(**self.save_file) 
            loss_dict = self.agent.policy.get_avg_loss()
            # wanb log
            train_info = dict(
                              reward=episode_score[-1], 
                              mean_q=np.mean(episode_q), 
                            )
            train_info.update(loss_dict)
            train_info.update(episode_reward_info)
            self.log_train(train_info, env_step)
                                        
            total_time = round(time.time() - time_start, 2)
            if episode % self.log_interval == 0:
                self.logger.info('phase: TRAIN, episodes: {}, '
                                 'episode_len: {}, episode reward: {:.2f}, '
                                 'mean_reward: {:.2f},'
                                 'q_loss: {:.6f}, total step: {}, total_time: {}, '
                                 'epsilon: {:.3f}, team_spirit:{:.3f}'.format(
                                episode, episode_step, episode_score[-1], mean_score.tolist(), 
                                loss_dict['q_loss'], env_step, total_time, 
                                self.agent.epsilon, self.env.team_spirit))
            # eval
            if self.use_eval:
                if episode % self.eval_interval == 0 or env_step >= self.total_train_step:
                    test_score = self.test(self.eval_episode, env_step)
                    if test_score > self.best_test_score:
                        self.best_test_score = test_score
                        save_file = {}
                        for key in self.save_file.keys():
                            save_file[key] = self.save_file[key].replace('.pth', f'_{env_step}.pth')
                        self.agent.save_weight(**save_file)
            episode += 1
        save_file = {}
        for key in self.save_file.keys():
            save_file[key] = self.save_file[key].replace('.pth', f'_final.pth')
        self.agent.save_weight(**save_file)
        self.logger.info('TRAINING DONE')

    def warmup(self):
        """
        随机动作回合
        """
        episode_score = []
        time_start = time.time() 
        env_step = 0
        episode = 0
        while env_step < self.total_warmup_step:
            episode_score.append(0)
            done = False
            obs = self.env.reset()
            episode_step = 0
            while not done:
                # todo
                if 'action_mask' in obs.keys():
                    action_mask = torch.Tensor(obs['action_mask']).bool()
                else:
                    action_mask = None
                all_actions = self.agent.random_sample(action_mask)
                obs_new, rewards, done, info, *_ = self.env.step(all_actions)
                episode_step +=1
                env_step += 1
                self.memory.push(obs, all_actions, rewards, obs_new, done)
                obs = obs_new
                episode_score[-1] += np.mean(rewards)
            episode += 1
            mean_score = np.mean(episode_score[-50:]) 
            total_time = round(time.time() - time_start, 2)
            
            self.logger.info('phase:WARMUP, episodes: {}, episode_len: {}, episode reward: {:.2f}, '
                             'mean_reward: {:.2f}, total_time: {}'.format(
                            episode, episode_step, episode_score[-1], mean_score.tolist(), total_time))
        self.logger.info('WARMUP DONE')

    def test(self, total_episode, env_step):
        
        """
        智能体测试回合
        """
        
        episode_reward_sequence = [[] for _ in range(self.n_eval_rollout_threads)]
        episode_sum_reward = []
        average_traking_ratio_sequence = []
        standard_deviation_sequence = []
        average_certainty_ratio_sequence = []
        average_collision_ratio_sequence = []
        obs = self.eval_envs.reset()
        episode_num = 0
        while episode_num < total_episode:
            actions = self.agent.select_actions(self.wrap_obs(obs),
                                                deterministic=True)
            actions = np.array(np.split(actions, self.n_eval_rollout_threads))
            obs,  rewards, dones, infos, reward_info = self.eval_envs.step(actions)
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

        info = f'eval_step: {env_step} '
        for key, val in evaluate_dict.items():
            info += f'{key}: {round(val,2)} '
        self.log_train(evaluate_dict, env_step)
        self.logger.info(info)
        if self.cfg.learning_stage == 0:
            test_score = np.mean(average_certainty_ratio_sequence)
        elif self.cfg.learning_stage == 1:
            test_score = np.mean(episode_sum_reward)
        else:
            test_score = np.mean(average_traking_ratio_sequence)
        return test_score

    def wrap_obs(self, obs:dict):
        wrapped_obs = {}
        for key, val in obs.items():
            wrapped_obs[key] = torch.from_numpy(np.concatenate(val)).to(dtype=torch.float32,device=self.device)
        return wrapped_obs
    
    def log_train(self, train_info, train_step):

        if self.use_wandb:
            for k, v in train_info.items():
                wandb.log({k:v}, step=train_step)
        else:
            for key, value in train_info.items():
                self.writer.add_scalar(key, value, train_step)
    
    def build_memory(self):

        self.mem_size = self.cfg.mem_size
        self.beta_min = self.cfg.beta
        self.alpha = self.cfg.alpha
        self.n_step = self.cfg.n_step
        self.batch_size = self.cfg.batch_size
        self.num_agents = self.env_info.agent_num
        obs_space = self.env.observation_space
        self.per = self.cfg.per
        if self.per:
            self.memory = MultiPrioritizedReplayBuffer(self.mem_size, self.batch_size,
                                                self.num_agents, obs_space, self.env_info['action_shape'],
                                                self.alpha, n_step=self.n_step, share_reward=self.cfg['share_reward'])  
        else:
            self.memory = MultiAgentReplayBuffer(self.mem_size, self.batch_size, obs_space,
                                                self.num_agents, self.env_info['action_shape'],
                                                n_step=self.n_step, share_reward=self.cfg['share_reward'])
                



