import torch.nn as nn
from .feature import FeatureNet
import numpy as np 
import torch.nn.functional as F
import torch 

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class PPOCritic(nn.Module):
    
    def __init__(self, config, env_info, device):
        super().__init__()

        self.feature_dim = config.feature_dim
        self.feature_net = FeatureNet(config, env_info)
        self._use_orthogonal = True
        self._gain = 0.1
        self.hidden_layer = mlp(self.feature_dim, config.hidden_dims, last_relu=True)
        self.value_net = nn.Linear(config.hidden_dims[-1], 1)
        self.to(device)
        
    def forward(self, obs):

        critic_feature = self.feature_net(obs)
        critic_feature = self.hidden_layer(critic_feature)
        value = self.value_net(critic_feature)
        
        return value
    
class PPOCentrialCritic(PPOCritic):
    
    def __init__(self, config, env_info, device):
        super().__init__(config, env_info, device)

        global_feature_dim = np.prod(env_info.global_info_shape)
        self.global_feature_net = nn.Linear(global_feature_dim, 256)
        self.hidden_layer = mlp(self.feature_dim+256, config.hidden_dims, last_relu=True)
        self.value_net = nn.Linear(config.hidden_dims[-1], 1)
        
        self.to(device)
        
    def forward(self, obs):

        local_critic_feature = self.feature_net(obs)
        global_obs = obs['global_info']
        global_critic_feature = F.relu(self.global_feature_net(global_obs)).reshape(-1, 256)
        full_feature = torch.cat([local_critic_feature, global_critic_feature], dim=1)
        critic_feature = self.hidden_layer(full_feature)
        value = self.value_net(critic_feature)
        
        return value