import torch.nn as nn
from algos.policy.utils.act import ACTLayer
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

class PPOActor(nn.Module):
    
    def __init__(self, config, env_info, device):

        super().__init__()
        self.feature_dim = config.feature_dim
        self.feature_net = FeatureNet(config, env_info)
        action_space = env_info.action_space
        self._use_orthogonal = True
        self._gain = 0.01
        self.hidden_layer = mlp(self.feature_dim, config.hidden_dims, last_relu=True)
        self.act = ACTLayer(action_space, config.hidden_dims[-1], self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, deterministic=False):

        actor_features = self.feature_net(obs)
        
        actor_features = self.hidden_layer(actor_features)

        actions, action_log_probs = self.act(actor_features, deterministic=deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, obs, action, active_masks_batch):
        actor_features = self.feature_net(obs)
        actor_features = self.hidden_layer(actor_features)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks = active_masks_batch)
        return action_log_probs, dist_entropy
    
    def get_action_logits(self, obs):
        
        actor_features = self.feature_net(obs)
        actor_features = self.hidden_layer(actor_features)
        action_logits = self.act.action_out(actor_features)
        
        return action_logits


        





    