import torch as T
import torch.nn as nn
from network.backbone.cnn import CNNLayer

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class FeatureNet(nn.Module):

    def __init__(self, config, env_info):
        
        super().__init__()
        agent_info_shape = env_info.agent_shape
        emb = config.emb_dim
        self.config = config
        self.use_matrix = env_info['use_matrix']
        if self.use_matrix:
            self.matrix_encoder = CNNLayer(config, env_info['matrix_shape'])
        self.friend_encoder = CNNLayer(config, env_info['local_friend_map_shape'])
        self.enermy_encoder = CNNLayer(config, env_info['local_target_map_shape'])
        self.agent_encoder = nn.Linear(agent_info_shape, emb)
        if self.use_matrix:
            input_dim = config['cnn_hidden_size']*3 + emb
        else: 
            input_dim = config['cnn_hidden_size']*2 + emb

        self.agg_net = nn.Linear(input_dim, config.feature_dim)

    def forward(self, input, i=None):
        """
        input shape : [batch, agent,*]
        output shape: [batch*agent, *]
        """    
        agent_obs = T.flatten(input['agent_info'], 0, 1)
        friend_obs = T.flatten(input['friend_info'], 0, 1)
        enermy_obs = T.flatten(input['target_info'], 0, 1)
        if self.use_matrix:
            matrix_obs = T.flatten(input['matrix_info'], 0, 1)

        friend_feature = self.friend_encoder(friend_obs)
        target_feature = self.enermy_encoder(enermy_obs)
        agent_feature = self.agent_encoder(agent_obs)
        if self.use_matrix:
            matrix_feature = self.matrix_encoder(matrix_obs)
            agg_feature = T.cat([agent_feature, friend_feature, target_feature, matrix_feature], dim=1)
        else:
            agg_feature = T.cat([agent_feature, friend_feature, target_feature], dim=1)
        state = self.agg_net(agg_feature)
        return state