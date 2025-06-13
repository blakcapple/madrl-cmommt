import torch.nn as nn
from network.backbone.cnn import CNNLayer
import torch as T
from network.backbone.atten import SelfAttention

class FeatureNet(nn.Module):
    
    def __init__(self, config, env_info):
        
        super().__init__()
        
        agent_info_shape = env_info.agent_shape
        emb = config.emb_dim
        self.config = config
        # self.position_embedding_dim = 10
        self.agent_encoder = nn.Linear(agent_info_shape, emb)
        self.friend_encoder = nn.Linear(agent_info_shape, emb)
        self.enermy_encoder = nn.Linear(agent_info_shape, emb)
        if config.feature_net == 'mlp':
            max_pursuer_in_obs = config.max_pursuer_in_obs
            max_evader_in_obs = config.max_evader_in_obs
            self.friend_attention = nn.Linear((max_pursuer_in_obs+1)*emb, emb)
            self.enermy_attention = nn.Linear((max_evader_in_obs+1)*emb, emb)
        elif config.feature_net == 'attention':
            self.friend_attention = SelfAttention(emb)
            self.enermy_attention = SelfAttention(emb)
        elif config.feature_net == 'mean_embedding':
            self.friend_fc = nn.Linear(2*emb, 3*emb)
            self.enermy_fc = nn.Linear(2*emb, 3*emb)
        if not config.feature_net == 'mean_embedding':
            self.friend_fc = nn.Linear(emb, 3*emb)
            self.enermy_fc = nn.Linear(emb, 3*emb)
        self.use_matrix = env_info['use_matrix']
        if self.use_matrix:
            matrix_shape = env_info['matrix_shape']
            self.matrix_encoder = CNNLayer(config, matrix_shape)
            input_dim = 6*emb+config['cnn_hidden_size']
        else:
            input_dim = 6*emb
            
        self.agg_net = nn.Linear(input_dim, config.feature_dim)
        
    
    def forward(self, input):
        # attention nedd position encoding

        agent_obs = T.flatten(input['agent_info'], 0, 1)
        friend_obs = T.flatten(input['friend_info'], 0, 1)
        enermy_obs = T.flatten(input['target_info'], 0, 1)
        friend_mask = T.flatten(input['friend_mask'], 0, 1)
        target_mask = T.flatten(input['target_mask'], 0, 1)
        if self.use_matrix:
            matrix_obs = T.flatten(input['matrix_info'], 0, 1)

        
        agent_emb = self.agent_encoder(agent_obs)
        friend_emb = self.friend_encoder(friend_obs)
        enermy_emb = self.enermy_encoder(enermy_obs)
        
        # friend info
        friend_input = T.cat([agent_emb.unsqueeze(1), friend_emb], dim=1)
        # apply mask
        friend_mask = T.cat([T.ones(friend_mask.shape[0],1).to(friend_mask.device), friend_mask], dim=1)
        if self.config.feature_net == 'mean_embedding':
            friend_feature = self.friend_fc(T.cat([agent_emb, T.mean(friend_emb, dim=1)], dim=1))
        elif self.config.feature_net == 'mlp':
            friend_feature = self.friend_fc(self.friend_attention(friend_input.view(friend_input.shape[0],-1)))
        elif self.config.feature_net == 'attention':
            friend_feature = self.friend_fc(self.friend_attention(friend_input, friend_mask)[:,0])
        
        # target info
        enermy_input = T.cat([agent_emb.unsqueeze(1), enermy_emb], dim=1)
        # apply mask
        target_mask = T.cat([T.ones(target_mask.shape[0],1).to(target_mask.device), target_mask], dim=1)
        if self.config.feature_net == 'mean_embedding':
            enermy_feature = self.enermy_fc(T.cat([agent_emb, T.mean(enermy_emb, dim=1)], dim=1))
        elif self.config.feature_net == 'mlp':
            enermy_feature = self.enermy_fc(self.enermy_attention(enermy_input.view(enermy_input.shape[0],-1)))
        elif self.config.feature_net == 'attention':
            enermy_feature = self.enermy_fc(self.enermy_attention(enermy_input, mask=target_mask))[:,0]
        
        if self.use_matrix:
            matrix_feature = self.matrix_encoder(matrix_obs)
            agg_feature = T.cat([friend_feature, enermy_feature, matrix_feature], dim=1)
        else:
            agg_feature = T.cat([friend_feature, enermy_feature], dim=1)
        state = self.agg_net(agg_feature)
        
        return state