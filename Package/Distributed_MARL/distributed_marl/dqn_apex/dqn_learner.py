from copy import deepcopy
import numpy as np
import ray
import random
import pdb
import shutil
import time
from distributed_marl.common.learner import ApexLearner
# from satenv.draw.animate import gif_plot
import time 
from torch.utils.tensorboard import SummaryWriter
from distributed_marl.algos.policy.iql import IQL
import pyarrow as pa
from distributed_marl.utils.util import set_seed
import os 

# @ray.remote
# def plot(update_step, save_path):
#     save_path = os.path.join(save_path, 'Update{}'.format(update_step))
#     os.makedirs(save_path, exist_ok=True)
#     shutil.copytree('env/draw/log', save_path+'/log', dirs_exist_ok=True) # copy log file to data dir
#     gif_plot(info_file=save_path+'/log/env_cfg.json', traj_file=save_path+'/log/trajs.xlsx', 
#     matrix_file=save_path+'/log/matrix.npy', plot_save_dir= save_path + os.path.normpath('/'))


@ray.remote(num_gpus=1)
class DQNLearner(ApexLearner):

    def __init__(self, brain, train_cfg: dict, env_cfg:dict):

        super().__init__(brain, train_cfg, env_cfg)
        self.device = self.cfg["learner_device"]
        self.brain = deepcopy(brain)
        self.policy = IQL(env_cfg, train_cfg, self.device, self.brain)
        self.param_update_interval = self.cfg['param_update_interval']
        self.max_num_updates = self.cfg['max_num_updates']
        self.save_interval = self.cfg['save_interval']
        self.batch_size = self.cfg['batch_size']
        set_seed(random.randint(1,999))

    def publish_params(self):
        new_params = self.get_params()
        new_params_id = pa.serialize(new_params).to_buffer()
        self.pub_socket.send(new_params_id)
        
    def publish_target_params(self):
        new_target_params = self.get_target_params()
        new_params_id = pa.serialize(new_target_params).to_buffer()
        self.pub_socket3.send(new_params_id)
        
    def recv_replay_data_(self):
        new_replay_data_id = self.learner_socket.recv()
        replay_data = pa.deserialize(new_replay_data_id)
        return replay_data
        
    def send_new_priorities(self, idxes: np.ndarray, priorities: np.ndarray):

        new_priors = [idxes, priorities]
        new_priors_id = pa.serialize(new_priors).to_buffer()
        self.learner_socket.send(new_priors_id)

    def get_params(self):

        params = []
        new_model = deepcopy(self.brain)
        state_dict = new_model.cpu().state_dict()
        for param in list(state_dict):
            params.append(state_dict[param].numpy())
        return params
    
    def get_target_params(self):
        
        params = []
        new_model = deepcopy(self.policy.target_mac)
        state_dict = new_model.cpu().state_dict()
        for param in list(state_dict):
            params.append(state_dict[param].numpy())
        return params
    
    def load_weights(self, load_dict: dict):
        if 'q_path' in load_dict.keys():
            self.policy.load_weights(load_dict['q_path'])

    def save_weights(self, save_dict:dict):
        if 'q_path' in save_dict.keys():
            self.policy.save_weights(save_dict['q_path'])
    
    def run(self):
        """
        leaner运行函数
        """
        try:
            time.sleep(1)
            print('learner starts running')
            log_dir = os.path.join(self.save_path, 'learner')
            self.logger = SummaryWriter(log_dir)
            self.update_step = 0
            s = time.time()
            while True:
                try:
                    replay_data = self.recv_replay_data_()
                    # new_priority, idxs = [], []
                    # for i in range(self.cfg["multiple_updates"]):
                    #     batch = dict()
                    #     for key, val in replay_data.items():
                    #         if key == 'state' or key == 'state2':
                    #             batch[key] = dict()
                    #             for inner_key, inner_val in val.items():
                    #                 batch[key][inner_key] = inner_val[self.batch_size*i:self.batch_size*(i+1)]
                    #         else:
                    #             batch[key] = val[self.batch_size*i:self.batch_size*(i+1)]
                    #     batch_new_priority, batch_idxs = self.policy.learn(deepcopy(batch), self.update_step)
                    #     new_priority.append(batch_new_priority)
                    #     idxs.append(batch_idxs)
                    # new_priority = np.concatenate(new_priority, axis=0)
                    # idxs = np.concatenate(idxs, axis=0)
                    for _ in range(self.cfg["multiple_updates"]):
                        new_priority, idxs = self.policy.learn(deepcopy(replay_data), self.update_step)
                    self.update_step = self.update_step + self.cfg["multiple_updates"]
                    self.send_new_priorities(idxs, new_priority)
                except Exception as e:
                    print('learner 更新网络时出现异常', e)

                if self.update_step % self.param_update_interval == 0:
                    self.publish_params()
                    self.publish_target_params()
                    loss = self.policy.get_avg_loss()
                    self.logger.add_scalar('TRAIN/loss', loss['q_loss'], self.update_step)
                # if self.update_step % 1000000 == 0:
                #     plot.remote(self.update_step*self.cfg["multiple_updates"], self.save_path)
                if self.update_step % self.save_interval == 0:
                    self.brain.save(self.save_path + f'/mac_{self.update_step}.pth')                    
                if self.update_step >= self.max_num_updates:
                    self.brain.save(self.save_path + f'/mac_final.pth')    
                    break
                if self.update_step % 1000 == 0 :
                    print(f'TOTAL UPDATA STEP:{self.update_step}, time:{time.time()-s}')
        except KeyboardInterrupt:
            import sys 
            sys.exit()



