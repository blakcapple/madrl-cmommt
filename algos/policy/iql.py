import torch as T 
import numpy as np 
import torch.nn.functional as F 
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
class IQL(object):
    '''
    Indepent Q-learning 
    '''
    def __init__(self, env_info, config, device, brain):
        
        self.gamma = config.gamma
        self.n_step = config.n_step
        self.update_target_interval = config.update_target_interval
        self.batch_size = config.batch_size
        self.num_agents = env_info.agent_num
        self.matrix_shape = env_info.matrix_shape
        self.action_space = env_info.action_space
        self.action_shape = env_info.action_shape
        self.per = config.per
        self.use_gradient_clip = config.use_gradient_clip
        if config.use_gradient_clip:
            self.gradient_clip = config.gradient_clip
        self.device = device
        self.mac = brain.to(self.device)
        self.target_mac = deepcopy(self.mac)
        self.loss = F.smooth_l1_loss
        self.prior_eps = 1e-6
        self.q_loss = []
        self.step = 0
        self.lr = config.lr 
        self.adam_eps = config.adam_eps
        self.optimizer = optim.Adam(self.mac.parameters(), self.lr, eps=self.adam_eps)
        self.use_rnn = config.use_rnn
        if self.use_rnn:
            self.ep_length = config.rnn_length

    def _compute_loss(self, experience, n_step=1):
        
        states, actions, rewards, states2, dones = experience
        actions = actions.long()
        rewards = rewards.view(-1)
        dones = dones.view(-1,1).expand(self.batch_size, self.num_agents).reshape(-1)
        # state_shape [batch, num_agent, state_dim]
        # action_shape [batch, num_agent]
        indices = np.arange(self.batch_size*self.num_agents)
        q1 = self.mac(states2) # [batch*num_agents, actions]
        if 'action_mask' in states2.keys():
            action_mask = T.flatten(states2['action_mask'].bool(), 0, 1)
            q1[action_mask] = -np.inf
        q2 = self.target_mac(states2)
        q_preds = self.mac(states)
        q_preds = q_preds[indices, actions.view(-1)]
        max_actions = T.argmax(q1, dim=1) # [batch*num_agents]
        q_targets = q2[indices, max_actions.view(-1)]
        targets = rewards + (self.gamma ** n_step)*q_targets*(1-dones)
        loss = self.loss(q_preds, targets, reduction='none').to(self.device)
        
        return loss
        
    def _compute_ep_loss(self, experience):
        states, actions, rewards, dones = experience
        hidden_state = self.mac.init_hidden(self.batch_size*self.num_agents)
        mac_out = []
        for i in range(self.ep_length+1):
            obs_in = {}
            for key, val in states.items():
                obs_in[key] = val[:,i]
            agent_outs, hidden_state = self.mac(obs_in, hidden_state)
            agent_outs = agent_outs.view(self.batch_size, self.num_agents, -1)
            mac_out.append(agent_outs)
        mac_out = T.stack(mac_out, dim=1)
        actions = actions.long()
        q_preds = T.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        target_mac_out = []
        hidden_state = self.target_mac.init_hidden(self.batch_size*self.num_agents)
        for i in range(self.ep_length+1):
            obs_in = {}
            for key, val in states.items():
                obs_in[key] = val[:,i]
            target_agent_outs, hidden_state = self.target_mac(obs_in, hidden_state)
            target_agent_outs = target_agent_outs.view(self.batch_size, self.num_agents, -1)
            target_mac_out.append(target_agent_outs)
        target_mac_out = T.stack(target_mac_out[1:], dim=1)
        mac_out_detach = mac_out.clone().detach()
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        cur_max_actions = cur_max_actions.long()
        q_targets = T.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        targets = rewards + self.gamma*q_targets*(1-dones[:,1:])
        loss = self.loss(q_preds, targets, reduction='none').to(self.device)

        return loss 

    def learn(self, data, update_step):
        state = {k:T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                    for k,v in data['state'].items()}
        action = T.as_tensor(data['act'], dtype=T.float32, device=self.device)
        reward = T.as_tensor(data['rew'], dtype=T.float32, device=self.device)
        done = T.as_tensor(data['done'], dtype=T.float32, device=self.device)
        if 'state2' in data.keys():
            state2 = {k:T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                    for k,v in data['state2'].items()}
            experience = (state, action, reward, state2, done)
        else:
            experience = (state, action, reward, done)
            
        if self.use_rnn:
            element_loss = self._compute_ep_loss(experience)
        else:
            element_loss = self._compute_loss(experience, self.n_step)
        element_loss = element_loss.reshape(self.batch_size, -1)
        # mask the invalid agent exp
        if 'agent_mask' in state.keys():
            agent_mask = state['agent_mask']
            element_loss = (element_loss*agent_mask.squeeze(2)).sum(dim=1) / agent_mask.sum(dim=1).squeeze(1)
        else:
            element_loss = T.mean(element_loss, dim=1)
        
        if self.per:
            weights = T.as_tensor(data['weights'], device=self.device)
            loss = T.mean(weights * element_loss)
        else:
            loss = T.mean(element_loss)
            
        self.q_loss.append(loss.item()) # store loss 
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_gradient_clip: 
            clip_grad_norm_(self.mac.parameters(), self.gradient_clip)
        self.optimizer.step()
            
        if update_step % self.update_target_interval == 0:
            self.target_mac = deepcopy(self.mac)
        if self.per:
            new_priority = element_loss.cpu().detach().numpy() + self.prior_eps
            idxs = data['idxs']
            return new_priority, idxs
        else:
            return None

    def save_weights(self, mac_path):
        self.mac.save(mac_path)

    def load_weights(self, mac_path):
        self.mac.load(mac_path, self.device)

    def update_network_parameters(self, tau):
        with T.no_grad():
            for q1, q1_tar in zip(self.mac.parameters(), self.target_mac.parameters()):
                q1_tar.data = q1_tar.data.mul(tau)
                q1_tar.data = q1_tar.data.add((1-tau)*q1.data)

    def get_avg_loss(self):
        assert(len(self.q_loss)) > 0
        critic_avg_loss = np.mean(self.q_loss)
        self.q_loss = []

        return dict(q_loss=critic_avg_loss)