from distributed_marl.common.architecture import Architecture
import ray 
import sys 

class DistributedPPO(Architecture):
    
    def __init__(self, worker_cls:type,
                       learner_cls:type,
                       port_cfg:dict,
                       config:dict,
                       common_cfg:dict,):
        super().__init__(worker_cls, learner_cls, config)

        self.config = config
        self.port_config = port_cfg
        self.common_cfg = common_cfg
    
    def spawn(self):

        self.workers = [
            self.worker_cls.remote(n, self.port_config, self.config, self.common_cfg)
            for n in range(1, self.num_workers+1)
        ]
        self.learner = self.learner_cls.remote(self.port_config, self.config, self.common_cfg)
        self.all_actors = self.workers + [self.learner]

    def train(self):

        id = ray.wait([actor.run.remote() for actor in self.all_actors])

        sys.exit()