import time
import numpy as np
import wandb
from utils.plot import plot_qloss, plot_rw
from algos.buffer.dqn_episode_buffer import MutiAgentEpisodeBuffer
from torch.utils.tensorboard import SummaryWriter
import os 
import torch

class RTrainer(object):
    '''Train process'''
    def __init__(self, env, env_info, agent, save_file, logger, log_file, train_config):
        self.env = env
        self.env_info = env_info
        self.agent = agent
        self.save_file = save_file
        self.use_wandb = train_config.use_wandb
        self.logger = logger
        self.log_file = log_file
        self.cfg = train_config
        self.total_train_step = train_config['train_step']
        self.total_warmup_step = train_config['warmup_step']
        self.log_path = train_config['save_path']
        self.use_eval =  train_config['use_eval']
        self.eval_interval = train_config['eval_interval']
        self.eval_episode = train_config['eval_episode']
        self.log_interval = train_config['log_interval']
        if not train_config['share_reward']:
            self.team_spirit = train_config['team_spirit']
            self.use_adaptive_team_spirit = train_config['use_adaptive_team_spirit']
            self.team_spirit_min = train_config['team_spirit_min']
            self.team_spirit_max = train_config['team_spirit_max']
            self.team_spirit_step = train_config['team_spirit_step']
        else:
            self.team_spirit = 1
        self.update_inteval = train_config['update_interval']
        self.total_update_step = int(self.total_train_step // self.update_inteval)
        if not self.use_wandb:
            log_dir = os.path.join(self.log_path, 'buffer')
            self.writer = SummaryWriter(log_dir)
        self.build_memory()
        self.update_step = 0
        self.best_score = -np.inf
        self.episode_length = 300

    def update(self):
        if self.memory.ready():
            data = self.memory.sample_batch()
            self.agent.train(data, self.update_step)
            self.update_step += 1

    def train(self):
        """
        智能体训练回合
        """
        episode_score = []
        time_start = time.time()
        env_step = 0
        episode = 0
        if not self.cfg['share_reward']:
            if self.use_adaptive_team_spirit:
                team_spirit = self.team_spirit_min + (self.team_spirit_max-self.team_spirit_min)* min(env_step / self.team_spirit_step, 1)
            else:
                team_spirit = self.team_spirit
        else:
            team_spirit = self.team_spirit
        obs = self.env.reset(team_spirit)
        hidden_state = self.agent.policy.mac.init_hidden(self.env.agent_num)
        done = False
        episode_score.append(0)
        episode_reward_info = {}
        episode_q = []
        env_episode = 0
        while env_step < self.total_train_step:
            # 加入第一个经验
            for key, val in obs.items():
                ep_idx = self.memory.ep_cntr // self.memory.capacity
                self.memory.state_buf[key][ep_idx][0] = val.copy()
            self.memory.done_buf[ep_idx][0] = done
            with torch.no_grad():
                for episode_step in range(self.rnn_lenth):
                    actions, q_value, hidden_state = self.agent.select_actions(obs, 'train', env_step, hidden_state)
                    obs_new, reward, done, info, reward_info = self.env.step(actions)
                    obs = obs_new
                    episode_score[-1] += np.mean(reward).item()
                    for key, val in reward_info.items():
                        episode_reward_info[key] = episode_reward_info.get(key, 0) + np.mean(val)
                    episode_q.append(q_value)
                    env_step +=1
                    if done:
                        if not self.cfg['share_reward']:
                            if self.use_adaptive_team_spirit:
                                team_spirit = self.team_spirit_min + (self.team_spirit_max-self.team_spirit_min)* min(env_step / self.team_spirit_step, 1)
                            else:
                                team_spirit = self.team_spirit
                        else:
                            team_spirit = self.team_spirit
                        obs = self.env.reset(team_spirit)
                        hidden_state = self.agent.policy.mac.init_hidden(self.env.agent_num)
                        if (env_episode+1) % 100 == 0 :
                            plot_rw(self.log_file, window_size=100, phase='train')
                            plot_qloss(self.log_file, window_size=100)
                        mean_score = np.mean(episode_score[-50:])
                        total_time = round(time.time() - time_start, 2)
                        if self.update_step >= 1:
                            mean_loss = self.agent.policy.get_avg_loss()['q_loss']
                            # wanb log
                            train_info = dict(loss=mean_loss, 
                                            reward=episode_score[-1], 
                                            mean_q=np.mean(episode_q), 
                                            )
                            train_info.update(episode_reward_info)
                            self.log_train(train_info, env_step)
                            self.logger.info('phase: TRAIN, episodes: {}, episode_len: {}, episode reward: {:.2f}, mean_reward: {:.2f}, best_reward: {:.2f}, q_loss: {:.6f}, total step: {}, total_time: {}, epsilon: {:.3f}, team_spirit:{:.3f}'.format(
                                    env_episode, self.episode_length, episode_score[-1], mean_score.tolist(), self.best_score, mean_loss, env_step, total_time, self.agent.epsilon, team_spirit))
                        else:
                            self.logger.info('phase: TRAIN, episodes: {}, episode_len: {}, episode reward: {:.2f}, mean_reward: {:.2f}, best_reward: {:.2f}, total step: {}, total_time: {}, epsilon: {:.3f}, team_spirit:{:.3f}'.format(
                                    env_episode, self.episode_length, episode_score[-1], mean_score.tolist(), self.best_score, env_step, total_time, self.agent.epsilon, team_spirit))
                        env_episode +=1 
                        episode_score.append(0)
                        episode_reward_info = {}
                        episode_q = []
                        if self.use_eval:
                            if env_episode % self.eval_interval == 0 or env_step >= self.total_train_step:
                                self.test(self.eval_episode, env_step)
                                obs = self.env.reset(team_spirit)
                                hidden_state = self.agent.policy.mac.init_hidden(self.env.agent_num)
                    self.memory.push(obs, actions, reward, done)             
            if episode % self.update_inteval == 0:
                self.update()
            episode += 1
            
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
            if not self.cfg['share_reward']:
                if self.use_adaptive_team_spirit:
                    team_spirit = self.team_spirit_min
                else:
                    team_spirit = self.team_spirit
            else:
                team_spirit = self.team_spirit
            obs = self.env.reset(team_spirit)
            episode_step = 0
            # 加入第一个经验
            for key, val in obs.items():
                self.memory.state_buf[key][self.memory.ep_cntr][0] = val.copy()
            self.memory.done_buf[self.memory.ep_cntr][0] = done
            while not done:
                all_actions = np.zeros((self.env.agent_num, 1), dtype=np.int)
                for i in range(self.env.agent_num):
                    action = self.env.action_space.sample()
                    all_actions[i] = action
                obs_new, rewards, done, info, *_ = self.env.step(all_actions)
                episode_step +=1
                env_step += 1
                obs = obs_new
                self.memory.push(obs, all_actions, rewards, done)
                episode_score[-1] += np.mean(rewards)
            episode += 1
            mean_score = np.mean(episode_score[-50:]) 
            total_time = round(time.time() - time_start, 2)
            
            self.logger.info('phase:WARMUP, episodes: {}, episode_len: {}, episode reward: {:.2f}, mean_reward: {:.2f}, total_time: {}'.format(
                         episode, episode_step, episode_score[-1], mean_score.tolist(), total_time))
        self.logger.info('WARMUP DONE')

    def test(self, episode_num, env_step):
        
        """
        智能体测试回合
        """
        episode_reward_sequence = []
        average_traking_ratio_sequence = []
        standard_deviation_sequence = []
        average_certainty_ratio_sequence = []
        average_collision_ratio_sequence = []
        for episode in range(episode_num):
            episode_reward = 0
            done = False
            obs = self.env.reset()
            episode_step = 0
            hidden_state = self.agent.policy.mac.init_hidden(self.env.agent_num)
            while not done:
                # 选择动作（此时的动作未乘上单步时长）
                actions, _, hidden_state = self.agent.select_actions(obs, 'test', hidden_state=hidden_state)
                # 执行动作 （此时的动作乘上单步时长）
                obs_new, reward, done, eval_infos, *_ = self.env.step(actions)
                episode_step +=1
                # 更新状态
                obs = obs_new
                episode_reward += np.mean(reward)
            episode_reward_sequence.append(episode_reward)
            average_traking_ratio_sequence.append(eval_infos.average_tracking_ratio)
            standard_deviation_sequence.append(eval_infos.standard_deviation)
            average_certainty_ratio_sequence.append(eval_infos.average_certainty_ratio)
            average_collision_ratio_sequence.append(eval_infos.average_collision_ratio)
        evaluate_dict = {'reward':np.mean(episode_reward_sequence),
                        'average_traking_ratio':np.mean(average_traking_ratio_sequence),
                        'standard_deviation':np.mean(standard_deviation_sequence),
                        'average_certainty_ratio':np.mean(average_certainty_ratio_sequence),
                        'average_collision_ratio':np.mean(average_collision_ratio_sequence),
                            }

        mean_score = np.mean(episode_reward_sequence)
        if self.best_score < mean_score:
            self.best_score = mean_score
            self.agent.save_weight(**self.save_file)
        self.log_train(evaluate_dict, env_step)
        log_info = f'phase: TEST, best_score: {round(self.best_score,2)} '
        for key, val in evaluate_dict.items():
            log_info += f'{key}: {round(val,2)} '
        self.logger.info(log_info)

    def log_train(self, train_info, train_step):
    
        if self.use_wandb:
            for k, v in train_info.items():
                wandb.log({k:v}, step=train_step)
        else:
            for key, value in train_info.items():
                self.writer.add_scalar(key, value, train_step)

    def build_memory(self):
    
        self.mem_size = self.cfg['mem_size']
        self.beta_min = self.cfg['beta']
        self.alpha = self.cfg['alpha']
        self.n_step = self.cfg['n_step']
        self.batch_size = self.cfg['batch_size']
        self.num_agents = self.env_info['pursuer_num']
        self.rnn_lenth = self.cfg['rnn_length']
        obs_space = self.env.observation_space
        self.memory = MutiAgentEpisodeBuffer(self.mem_size, self.rnn_lenth, self.batch_size,
                                                obs_space, self.num_agents, self.env_info['action_shape'],
                                                share_reward=self.cfg['share_reward'])
 
                


