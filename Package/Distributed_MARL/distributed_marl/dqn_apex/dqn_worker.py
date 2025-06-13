import numpy as np
import ray
import torch
import torch.nn as nn
import time 
from distributed_marl.common.worker import ApeXWorker
from distributed_marl.algos.utils.epsilon_decay import DecayThenFlatSchedule
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torch.nn.functional as F
import time
import os 
from gym import spaces
# smmoth average
def running_mean(x,n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    if len(cumsum) < n+1:
        n = len(cumsum)-1
    return (cumsum[n:] - cumsum[:-n]) / float(n)

@ray.remote
class DQNWorker(ApeXWorker):

    def __init__(
        self, worker_id: int, worker_brain: nn.Module, train_cfg: dict, env_info, env
    ):  
        super().__init__(worker_id, worker_brain, train_cfg, env)
        self.use_wandb = train_cfg["use_wandb"]
        # set policy
        self.device = train_cfg["worker_device"]
        # read necessary infomation
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.gamma = self.cfg["gamma"]
        self.num_step = self.cfg["n_step"]
        self.action_shape = env_info['action_shape']
        # set necessary parameters
        self.nstep_queue = deque(maxlen=self.num_step)
        self.epsilon = 0.4**(1+(worker_id-1)*7/(self.cfg['num_workers']-1))
        self.episode = 0
        self.prior_eps = 1e-6
        self.num_agents = env_info.agent_num
        self.env_step = 0

        self.save_path = self.cfg['save_path']

    def preprocess_data(self, nstepqueue: deque):
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)
        with torch.no_grad():
            new_priority = self.compute_q_loss(nstep_data)

        return nstep_data, new_priority
        
    def compute_q_loss(self, data):

        state, action, reward, state2, done = data
        state = {k:torch.tensor([v], dtype=torch.float32, device=self.device) for k,v in state.items()}
        state2 = {k:torch.tensor([v], dtype=torch.float32, device=self.device) for k,v in state2.items()}
        q_value = self.brain(state)
        q_target_value2 = self.target_brain(state2)
        q_value = q_value[range(q_value.shape[0]), action.reshape(-1)]
        state2_value = self.brain(state2)
        if 'action_mask' in state2.keys():
            action_mask = torch.flatten(state2['action_mask'], 0, 1).bool()
            state2_value[action_mask] = -np.inf
        
        max_actions = torch.argmax(state2_value, dim=1)
        target_q = q_target_value2[range(q_value.shape[0]),max_actions.view(-1)]
        target_q_value = self.gamma ** self.num_step * target_q * (1-done) + reward
        priority_value = F.smooth_l1_loss(target_q_value, q_value)
        priority_value = priority_value.item()
        new_priority = priority_value + self.prior_eps

        return new_priority

    def collect_data(self):
        local_buffer = []
        nstep_queue = deque(maxlen=self.num_step)
        epret_squence = []
        while len(local_buffer) < self.worker_buffer_size:
            action = self.select_action(self.state)
            next_state, reward, done, *_ = self.env.step(action)
            nstep_queue.append((self.state, action, reward, next_state, done))
            self.episode_reward += np.mean(reward)
            if (len(nstep_queue) == self.num_step) or done:
                nstep_data, priority_value = self.preprocess_data(nstep_queue)
                local_buffer.append([nstep_data, priority_value])
            if done:
                epret_squence.append(self.episode_reward)
                self.state = self.env.reset()
                self.logger.add_scalar('TRAIN/worker{}_reward'.format(self.worker_id), self.episode_reward, self.episode-1)
                self.episode_reward = 0
                self.episode +=1
            else:
                self.state = next_state
        return local_buffer

    def select_action(self, state, deterministic=False):

        self.env_step +=1
        state = {k:torch.tensor([v], dtype=torch.float32, device=self.device) for k,v in state.items()}
        if 'action_mask' in state.keys():
            action_mask = state['action_mask'][0].bool()
        else:
            action_mask = None
        if deterministic:
            values = self.brain(state)
            if action_mask is not None:
                values[action_mask] = -np.inf
            actions = torch.argmax(values, dim=1).tolist()
        else:
            if np.random.random() > self.epsilon:
                values = self.brain(state)
                if action_mask is not None:
                    values[action_mask] = -np.inf
                actions = torch.argmax(values, dim=1).tolist()
            else:
                if action_mask is not None:
                    action_length = len(action_mask[0])
                    action_space = []
                    for i in range(action_mask.shape[0]):
                        action_space.append(spaces.Discrete(action_length - sum(action_mask[i]).cpu().numpy()))
                    actions = [action_space[i].sample() for i in range(self.num_agents)] 
                else:
                    actions = [self.env.action_space.sample() for i in range(self.num_agents)] 
        actions = np.array(actions).reshape(-1, self.action_shape)
        return actions

    def act(self):
        try:
            local_buffer = self.collect_data() # 收集数据 直到local buffer 满
            self.send_replay_data(local_buffer) # 向global buffer 发送 local buffer 的数据
            self.receive_new_params() # 接收新的model参数
            self.receive_target_new_params()
        except Exception as e:
            print(f'worker {self.worker_id} 出现异常')

    def run(self):
        try:
            log_dir = os.path.join(self.save_path, 'worker')
            self.logger = SummaryWriter(log_dir)
            print(f'worker {self.worker_id} starts running')
            if self.worker_id == 1:
                """
                evaluate
                """
                self.test_run()
            else:
                self.episode_reward = 0
                self.state = self.env.reset()
                while True:
                    self.act()
        except KeyboardInterrupt:
            import sys
            sys.exit()


    def test_run(self):
        self.env.set_render_flag()
        best_reward = -np.inf
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
                state = self.env.reset()
                done = False
                env_step = 0 
                episode_reward_info = {}
                while True:
                    action = self.select_action(state, True)
                    next_state, reward, done, info, reward_info = self.env.step(action)
                    episode_reward += np.mean(reward)
                    for key, val in reward_info.items():
                        episode_reward_info[key] = episode_reward_info.get(key, 0) + np.mean(val)
                    state = next_state
                    env_step +=1
                    if done:
                        episode_reward_sequence.append(episode_reward)
                        average_traking_ratio_sequence.append(info.average_tracking_ratio)
                        standard_deviation_sequence.append(info.standard_deviation)
                        average_certainty_ratio_sequence.append(info.average_certainty_ratio)
                        average_collision_ratio_sequence.append(info.average_collision_ratio)
                        episode +=1
                        break
                mean_reward = np.mean(episode_reward_sequence[-100:])
                evaluate_dict = {'reward':episode_reward,
                                'average_traking_ratio':info.average_tracking_ratio,
                                'standard_deviation':info.standard_deviation,
                                'average_certainty_ratio':info.average_certainty_ratio,
                                'average_collision_ratio':info.average_collision_ratio,
                                 }
                evaluate_dict.update(episode_reward_info)
                self.send_evaluate_data(evaluate_dict)
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    save_env_log_dir = os.path.join(self.save_path, 'env_logs')
                    if not os.path.exists(save_env_log_dir):
                        os.mkdir(save_env_log_dir)
                    # self.env.save_info(save_env_log_dir)
                    self.brain.save(self.save_path + f'/mac_{self.env.get_stage()}.pth')
                # total_time = time.time() - time_start
                # print('phase: TRAIN, episodes: {}, episode_len: {}, episode reward: {:.2f}, mean_reward: {:.2f}, best_reward: {:.2f}, total_time:{:.2f}, stage:{}'.format(episode, env_step, episode_reward, mean_reward, best_reward, total_time, self.env.get_stage()))
            else:
                pass
