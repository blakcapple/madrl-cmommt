from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import pyarrow as pa
import torch.nn as nn
import zmq

class ApexLearner(ABC):
    def __init__(
        self, brain: nn.Module, train_cfg: dict, env_cfg: dict
    ):
        self.cfg = train_cfg

        # unpack communication configs
        self.pairport_tolearner = train_cfg["pairport_tolearner"]
        self.pubsub_port = train_cfg["pubsub_port"]
        self.pubsub_port3 = train_cfg["pubsub_port3"]

        # initialize zmq sockets
        print("[Learner]: initializing sockets..")
        self.initialize_sockets()
        self.save_path = self.cfg['save_path']


    @abstractmethod
    def get_params(self):
        """Return model params for synchronization"""
        pass

    @abstractmethod
    def load_weights(self, load_dict:dict):
        """
        载入预训练的模型参数
        """
        pass
    
    @abstractmethod
    def save_weights(self, save_dict:dict):
        """
        保存训练中的模型参数
        """
        pass

    @abstractmethod
    def publish_params(self, new_params: np.ndarray):
        """
        发布模型参数
        """

    @abstractmethod
    def recv_replay_data_(self):
        """
        接收replay data到缓存队列
        """
        
    @abstractmethod
    def send_new_priorities(self, idxes: np.ndarray, priorities: np.ndarray):
        """
        发布replay data新的优先级
        """

    @abstractmethod
    def run(self):
        """
        主运行函数
        """

    def initialize_sockets(self):
        """
        初始话通信接口
        """
        # For sending new params to workers
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.pubsub_port}")
        # this is for sending target params
        self.pub_socket3 = context.socket(zmq.PUB)
        self.pub_socket3.bind(f"tcp://127.0.0.1:{self.pubsub_port3}")

        # For receiving batch from, sending new priorities to Buffer # write another with PUSH/PULL for non PER version
        context = zmq.Context()
        self.learner_socket = context.socket(zmq.PAIR)
        self.learner_socket.bind(f"tcp://127.0.0.1:{self.pairport_tolearner}")




    


