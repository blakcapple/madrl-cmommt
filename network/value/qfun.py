import numpy as np 
import torch as T
import torch.nn as nn
from .feature import FeatureNet
from .map_feature import FeatureNet as MapFeatureNet
from network.backbone.rnn import RNNLayer

# position encoding 
def getPositionEncoding(seq_len, d, n=100):
    P = T.zeros(seq_len, d)
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class Qfun(nn.Module):
    
    def __init__(self, config, env_info):
        
        super().__init__()
        self.feature_dim = config.feature_dim
        self.a_dim = env_info.action_dim 
        self.q_hidden_dims = config.hidden_dims
        if config.use_map_obs:
            self.feature_net = MapFeatureNet(config, env_info)
        else:
            self.feature_net = FeatureNet(config, env_info)
        self.use_rnn = config.use_rnn
        if not self.use_rnn:
            self.q_layer = mlp(self.feature_dim, self.q_hidden_dims+[self.a_dim])
        else:
            self.q_layer = RNNLayer(self.feature_dim, config)
            self.action_layer = nn.Linear(config.rnn_hidden_dim, self.a_dim)
            
    def init_hidden(self, hidden_size):
        # make hidden states on same device as model
        hidden_state = self.q_layer.init_hidden(hidden_size)
        return hidden_state
    
    def forward(self, obs, hidden_state=None):
        feature = self.feature_net(obs)
        if not self.use_rnn:
            q = self.q_layer(feature)
            return q
        else:
            h = self.q_layer(feature, hidden_state)
            q = self.action_layer(h)
            return q, h
    
    def save(self, path):
        T.save(self.state_dict(), path, _use_new_zipfile_serialization=False)
    
    def load(self, path, device='cpu'):
        self.load_state_dict(T.load(path, map_location=device), strict=False)

