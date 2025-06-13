import numpy as np 
import torch as T
import torch.nn as nn
import torch.optim as optim

class QMixer(nn.Module):
    """
    Q-mix network for marl
    input shape: [batch, ep_length, *]
    output shape: [batch, ep_length, 1]
    """
    def __init__(self, config, env_config):
        super(QMixer,self).__init__()
        self.embed_dim = config.qmix_embed_dim
        self.n_agents = env_config.agent_num
        self.state_dim =  np.prod(env_config.global_info_shape)
        self.hyper_w1 = nn.Linear(self.state_dim, self.n_agents*self.embed_dim)
        self.hyper_w2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b2 =  nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
    
    def forward(self, agent_qs, global_info):

        global_info = global_info.reshape(-1, self.state_dim)
        bs = agent_qs.size(0) 
        # first layer 
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        w1 = T.abs(self.hyper_w1(global_info))
        b1 = self.hyper_b1(global_info)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = T.nn.functional.elu(T.bmm(agent_qs, w1) + b1)

        # second layer 
        w2 = T.abs(self.hyper_w2(global_info))
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(global_info).view(-1,1,1)
        final = T.bmm(hidden, w2) + b2
        q_total = final.view(bs, -1, 1)

        return q_total

    def save(self, path):
        T.save(self.state_dict(), path, _use_new_zipfile_serialization=False)
    
    def load(self, path):
        self.load_state_dict(T.load(path))




    
