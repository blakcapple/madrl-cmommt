from typing import Deque, Union
import ray
import torch.nn as nn
from distributed_marl.common.architecture import Architecture
from distributed_marl.dqn_apex.buffer_helper import PrioritizedReplayBufferHelper
from distributed_marl.dqn_apex.learner_helper import LearnerHelper

class ApeX(Architecture):
    def __init__(
        self,
        worker_cls: type,
        learner_cls: type,
        brain: Union[tuple, nn.Module],
        env_info:dict,
        all_config:dict,
        obs_space, 
        set_env,
    ):
        self.all_cfg = all_config
        self.env_info = env_info
        self.common_cfg = all_config
        super().__init__(worker_cls, learner_cls, self.all_cfg)

        self.brain = brain
        if type(brain) is tuple:
            self.worker_brain = self.brain[0]
        else:
            self.worker_brain = self.brain
            
        self.obs_space = obs_space
        self.env_fun = set_env

    def spawn(self):
        # Spawn all components
        env_list = [self.env_fun(self.all_cfg, n) for n in range(1, self.num_workers+1)]
        self.learner = self.learner_cls.remote(self.brain, self.all_cfg, self.env_info)
        self.global_buffer = PrioritizedReplayBufferHelper.remote(self.all_cfg, self.env_info, self.obs_space)
        self.leaner_helper = LearnerHelper.remote(self.all_cfg)
        self.workers = [
            self.worker_cls.remote(n, self.worker_brain, self.all_cfg, self.env_info, env_list[n-1])
            for n in range(1, self.num_workers + 1)
        ]
        self.all_actors = self.workers + [self.learner] + [self.global_buffer] + [self.leaner_helper]

    def train(self):
        # TODO: implement a safer exit
        print("Running main training loop...")
        id = ray.wait([actor.run.remote() for actor in self.all_actors])
        # sys.exit()
        # while True:
        #     pass  