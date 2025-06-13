import torch.nn as nn
from network.backbone.cnn import CNNLayer
import torch as T
from network.backbone.atten import SelfAttention

class FeatureNet(nn.Module):
    
    def __init__(self, config, env_info):
        
        super().__init__()
        
        agent_info_shape = env_info.agent_shape
        emb = config.emb_dim
        # self.position_embedding_dim = 10
        self.agent_encoder = nn.Linear(agent_info_shape, emb)
        self.friend_encoder = nn.Linear(agent_info_shape, emb)
        self.enermy_encoder = nn.Linear(agent_info_shape, emb)
        self.friend_attention = SelfAttention(emb)
        self.enermy_attention = SelfAttention(emb)
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

        agent_obs = input['agent_info']
        friend_obs = input['friend_info']
        enermy_obs = input['target_info']
        friend_mask = input['friend_mask']
        target_mask = input['target_mask']
        if self.use_matrix:
            matrix_obs = input['matrix_info']
        
        agent_emb = self.agent_encoder(agent_obs)
        friend_emb = self.friend_encoder(friend_obs)
        enermy_emb = self.enermy_encoder(enermy_obs)
        # friend info
        friend_input = T.cat([agent_emb.unsqueeze(1), friend_emb], dim=1)
        # apply mask
        friend_mask = T.cat([T.ones(friend_mask.shape[0],1).to(friend_mask.device), friend_mask], dim=1)
        friend_feature = self.friend_fc(self.friend_attention(friend_input, friend_mask)[:,0])
        
        # target info
        enermy_input = T.cat([agent_emb.unsqueeze(1), enermy_emb], dim=1)
        # apply mask
        target_mask = T.cat([T.ones(target_mask.shape[0],1).to(target_mask.device), target_mask], dim=1)
        enermy_feature = self.enermy_fc(self.enermy_attention(enermy_input, mask=target_mask))[:,0]
        
        if self.use_matrix:
            matrix_feature = self.matrix_encoder(matrix_obs)
            agg_feature = T.cat([friend_feature, enermy_feature, matrix_feature], dim=1)
        else:
            agg_feature = T.cat([friend_feature, enermy_feature], dim=1)
        state = self.agg_net(agg_feature)
        
        return state