from copy import deepcopy
import time
from abc import ABC, abstractmethod
import pyarrow as pa
import torch
import torch.nn as nn
import zmq
from distributed_marl.utils.util import set_seed
# from satenv.base import BaseEnv
# from satenv.configs.base_cfg import env_config
# from satenv.wrapper.meta_env import MetaEnv

class Worker(ABC):
    """Abstract class for ApeX distrbuted workers """
    def __init__(
        self, worker_id: int, train_cfg: dict, 
    ):
        self.worker_id = worker_id
        self.cfg = train_cfg
        # unpack communication configs
        self.pubsub_port = train_cfg["pubsub_port"]
        self.pubsub_port3 = train_cfg['pubsub_port3']
        self.pullpush_port = train_cfg["pullpush_port"]
        if self.worker_id == 1:
            self.pair_port = train_cfg['pair_port'] # send evaluate data to global buffer 
        if self.worker_id != 1:
            self.sub_port = train_cfg['pubsub_port2']
        # initialize zmq sockets
        print(f"[Worker {self.worker_id}]: initializing sockets..")
        self.initialize_sockets()

        self.learning_stage = 0

    @abstractmethod
    def select_action(self, state) :
        """Select action with worker's brain"""
        pass

    @abstractmethod
    def preprocess_data(self, data):
        """Preprocess collected data if necessary (e.g. n-step)"""
        pass

    @abstractmethod
    def collect_data(self):
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    @abstractmethod
    def test_run(self):
        """Specifically for the performance-testing worker"""
        pass

    @abstractmethod
    def run(self):
        """main runing loop"""
        pass

    @abstractmethod    
    def synchronize(self, new_params: list):
        """Synchronize worker brain with parameter server"""
        pass 

    @abstractmethod
    def send_replay_data(self, replay_data):
        """
        send collected data to central buffer
        """
    
    @abstractmethod
    def receive_new_params(self):
        """
        receive new params from learner
        """

    def initialize_sockets(self):
        # for receiving params from learner
        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB) # 订阅模式
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅模式订阅的话题
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.pubsub_port}")
        # for target_q
        self.sub_socket3 = context.socket(zmq.SUB) # 订阅模式
        self.sub_socket3.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅模式订阅的话题
        self.sub_socket3.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket3.connect(f"tcp://127.0.0.1:{self.pubsub_port3}")

        # for sending replay data to buffer
        time.sleep(1)
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.pullpush_port}")

        if self.worker_id == 1:
            context = zmq.Context()
            self.pair_socket = context.socket(zmq.PAIR)
            self.pair_socket.bind(f"tcp://127.0.0.1:{self.pair_port}")

        if self.worker_id != 1:
            self.sub_socket_2 = context.socket(zmq.SUB) # 订阅模式
            self.sub_socket_2.setsockopt_string(zmq.SUBSCRIBE, "stage") # 订阅模式订阅的话题
            # self.sub_socket_2.setsockopt(zmq.CONFLATE, 1)
            self.sub_socket_2.connect(f"tcp://127.0.0.1:{self.sub_port}")

class ApeXWorker(Worker):

    def __init__(
        self, worker_id: int, brain: nn.Module, train_cfg: dict, env
        ):
        super().__init__(worker_id, train_cfg)

        self.brain = deepcopy(brain)
        self.target_brain = deepcopy(self.brain)
        self.seed = worker_id * 1000
        # create env
        self.env = env
        set_seed(self.seed)
        self.save_path = train_cfg['save_path']

    def synchronize(self, new_params: list):
        
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param.copy()).to(self.device)
            param.data.copy_(new_param)
            
    def synchronize_target(self, new_params: list):
        
        for param, new_param in zip(self.target_brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param.copy()).to(self.device)
            param.data.copy_(new_param)
            
    def send_replay_data(self, replay_data):
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data_id)

    def receive_new_params(self):
        new_params_id = False
        try:
            new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_params_id:
            new_params = pa.deserialize(new_params_id)
            self.synchronize(new_params)
            return True
        
    def receive_target_new_params(self):

        new_params_id = False
        try:
            new_params_id = self.sub_socket3.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_params_id:

            new_params = pa.deserialize(new_params_id)
            self.synchronize_target(new_params)
            return True

    def receive_new_learning_stage(self):

        new_stage_id = False 
        try:
            new_stage_id = self.sub_socket_2.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if new_stage_id:
            topic, messagedata = new_stage_id.split()
            self.learning_stage = int(messagedata)

    def send_evaluate_data(self, evaluate_data):
        evaluate_data_id = pa.serialize(evaluate_data).to_buffer()
        self.pair_socket.send(evaluate_data_id)



