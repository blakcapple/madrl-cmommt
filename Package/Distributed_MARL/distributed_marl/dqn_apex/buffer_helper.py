from pathlib import Path
import pyarrow as pa
import ray
import zmq
import pdb
from distributed_marl.algos.utils.replay_buffer import MultiPrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import wandb 
import time
import os

@ray.remote
class PrioritizedReplayBufferHelper(object):
    def __init__(self, buffer_cfg: dict, env_info:dict, obs_space):
        self.cfg = buffer_cfg
        # unpack buffer configs
        self.max_num_updates = self.cfg["max_num_updates"]
        self.priority_alpha = self.cfg["alpha"]
        self.priority_beta = self.cfg["beta"]
        self.priority_beta_increment = (
            1 - self.priority_beta
        ) / self.max_num_updates

        self.sample_size = self.cfg['sample_size']
        self.mem_size = self.cfg['mem_size']

        self.num_agents = env_info.agent_num
        self.action_shape = env_info.action_shape
        self.buffer = MultiPrioritizedReplayBuffer(self.mem_size, self.sample_size,
                                                self.num_agents, obs_space, self.action_shape,
                                                self.priority_alpha) 

        # unpack communication configs
        self.pairport_tobuffer = self.cfg["pairport_tobuffer"]
        self.pullpush_port = self.cfg["pullpush_port"]
        # worker pub and buffer sub
        self.pair_port = self.cfg['pair_port']
        self.pub_port = self.cfg['pubsub_port2']
        

        # initialize zmq sockets
        print("[Buffer]: initializing sockets..")
        self.initialize_sockets()
        self.update_counter = 0 
        self.sample_counter = 0

        self.evaluate_dict = {}
        self.use_wandb = self.cfg["use_wandb"]
        self.save_path = self.cfg['save_path']

        self.consuming_rate = self.cfg.consuming_rate

    def initialize_sockets(self):
        # for sending batch to learner and retrieving new priorities
        context = zmq.Context()
        self.buffer_socket = context.socket(zmq.PAIR)
        self.buffer_socket.connect(f"tcp://127.0.0.1:{self.pairport_tobuffer}")

        # for receiving replay data from workers
        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.pullpush_port}")

        # for receiving reward and learning stage information from worker1;
        context = zmq.Context()
        self.pair_socket = context.socket(zmq.PAIR)
        self.pair_socket.connect(f"tcp://127.0.0.1:{self.pair_port}")

        # for sending learning stage to other worker
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.pub_port}")

    def send_batch(self):
        # send batch and request priorities (blocking recv)
        # 保持读写比为给定的常数
        self.consuming_rate =  (self.update_counter*self.sample_size) // self.buffer.mem_cntr
        if self.consuming_rate > self.cfg.consuming_rate:
            time.sleep(3)
        if self.sample_counter - self.update_counter < 100:
            batch = self.buffer.sample_batch(self.priority_beta)
            batch_id = pa.serialize(batch).to_buffer()
            self.buffer_socket.send(batch_id)
            self.priority_beta += self.priority_beta_increment
            self.sample_counter +=1

    def recv_priors(self):
        # receive and update priorities
        new_priors_id = False
        try:
            new_priors_id = self.buffer_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass 
        if new_priors_id:
            idxes, new_priorities = pa.deserialize(new_priors_id)
            self.buffer.update_priorities(idxes, new_priorities)
            self.update_counter +=1

    def recv_data(self):
        new_replay_data_id = False
        try:
            new_replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass

        if new_replay_data_id:
            new_replay_data = pa.deserialize(new_replay_data_id)
            for replay_data, priority in new_replay_data:
                self.buffer.push(*replay_data)
                self.buffer.update_priorities([(self.buffer.mem_cntr-1)%self.mem_size], [priority])

    def recv_evaluate_data(self):
        new_evaluate_data_id = False
        try: 
            new_evaluate_data_id = self.pair_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if new_evaluate_data_id:
            new_data = pa.deserialize(new_evaluate_data_id)
            self.evaluate_dict = new_data

            
    def run(self):
        try:
            print('buffer starts running')
            if self.use_wandb:
                wandb.init(project="Pursue-Evation", entity="the-one")
            else:
                log_dir = os.path.join(self.save_path, 'buffer')
                self.logger = SummaryWriter(log_dir)
            start_time_1 = time.time()
            start_time = time.time()
            while True:
                try:
                    self.recv_data() # 从worker接收data
                    self.recv_evaluate_data() # 从worker1接收evaluate data
                except Exception as e:
                    print('buffer 接收数据时出现异常', e)
                if len(self.buffer) > self.sample_size:
                    self.send_batch()
                    self.recv_priors()
                    
                else:
                    pass

                if time.time() - start_time > 10:# 每隔60秒记录一次

                    time_elapse = int(time.time() - start_time_1)
                    if not self.use_wandb:
                        self.logger.add_scalar('Buffer/size', self.buffer.mem_cntr, time_elapse)
                    else:
                        wandb.log({'time':time_elapse},step=self.buffer.mem_cntr)
                    start_time = time.time()
                    if self.evaluate_dict:
                        self.log_info(self.buffer.mem_cntr)
                        running_fps = self.buffer.mem_cntr / time_elapse
                        reward = self.evaluate_dict['reward']
                        print(f'fps:{running_fps}, reward:{reward}, consuming_rate: {self.consuming_rate}')
                        self.evaluate_dict = {}
        except KeyboardInterrupt:
            import sys 
            sys.exit()

    def log_info(self, step):
        if self.use_wandb:
            for key, value in self.evaluate_dict.items():
                wandb.log({key:value}, step)
        else: 
            for key, value in self.evaluate_dict.items():
                self.logger.add_scalar(key, value, step)
    
    def send_learning_stage_data(self, data):

        self.pub_socket.send_string(f"stage {data}")



        
            