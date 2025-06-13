import torch as T 
import numpy as np 
import torch.nn.functional as F 
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_

class QMix(object):
    """
    for Q-mix training
    params: mac(multiagent controller) -> deep network
            env_config 环境参数
            config 参数
    """
    def __init__(self, env_config, config, device, brain, mixer, use_rnn):

        self.gamma = config['gamma']
        self.n_step = config['n_step']
        self.batch_size = config['batch_size']
        self.num_agents = env_config['num_dan']
        self.update_target_step = config['update_target_step']
        self.use_rnn = use_rnn
        self.device = device
        self.mac = brain.to(self.device)
        self.target_mac = deepcopy(self.mac)
        self.mixer = mixer.to(self.device)
        self.target_mixer = deepcopy(self.mixer)
        self.use_gradient_clip = config['use_gradient_clip']
        if config['use_gradient_clip']:
            self.gradient_clip = config["gradient_clip"]
        if not use_rnn:
            self.loss = F.smooth_l1_loss
        else:
            self.ep_length = config['rnn_length']
            self.loss = F.mse_loss
        self.prior_eps = 1e-6
        self.q_loss = []
        self.update_step = 0

    def _compute_q_loss(self, data, n_step=1):

        states, actions, rewards, states2,dones = data['state'], data['act'], \
                                       data['rew'], data['state2'], data['done']
        actions = actions.long()
        # state_shape [batch, num_agent, state_dim]
        # action_shape [batch, num_agent]
        indices = np.arange(self.batch_size*self.num_agents)
        q1 = self.mac(states2) # [batch*num_agents, actions]
        q2 = self.target_mac(states2)
        q_pred = self.mac(states)
        q_preds = q_pred[indices, actions.view(-1)]
        max_actions = T.argmax(q1, dim=1) # [batch*num_agents]
        q_targets = q2[indices, max_actions.view(-1)]
        q_total = self.mixer(q_preds.view(-1,self.num_agents), states['global_info']).reshape(-1) # [batch,1,1]
        q_targets_total = self.target_mixer(q_targets.view(-1, self.num_agents), states2['global_info']).reshape(-1)
        targets = rewards.view(-1) + (self.gamma ** n_step)*q_targets_total*(1-dones)
        loss = self.loss(q_total, targets, reduction='none').to(self.device)

        return loss 

    def _compute_ep_loss(self, data):

        states, actions, rewards, dones = data['state'], data['act'][:,:-1], \
                                        data['rew'][:,:-1], data['done'][:,:-1]
        self.mac.init_hidden(self.batch_size*self.num_agents)
        mac_out = []
        for i in range(self.ep_length):
            agent_outs = self.mac(states, i)
            agent_outs = agent_outs.view(self.batch_size, self.num_agents, -1)
            mac_out.append(agent_outs)
        mac_out = T.stack(mac_out, dim=1)
        q_preds = T.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(self.batch_size*self.num_agents)
        for i in range(self.ep_length):
            target_agent_outs = self.target_mac(states, i)
            target_agent_outs = target_agent_outs.view(self.batch_size, self.num_agents, -1)
            target_mac_out.append(target_agent_outs)
        target_mac_out = T.stack(target_mac_out[1:], dim=1)
        mac_out_detach = mac_out.clone().detach()
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        q_targets = T.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        q_preds = self.mixer(q_preds, states['global_info'][:,:-1])
        q_targets = self.target_mixer(q_targets, states['global_info'][:,1:])
        targets = rewards + self.gamma*q_targets*(1-dones)
        loss = self.loss(q_preds, targets).to(self.device)

        return loss 

    def learn(self, data, update_step):

        if not self.use_rnn:
            data['state'] = {k:T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                for k,v in data['state'].items()}
            data['state2'] = {k:T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                for k,v in data['state2'].items()}
            data['act'] = T.as_tensor(data['act'], dtype=T.float32, device=self.device)
            data['rew'] = T.as_tensor(data['rew'], dtype=T.float32, device=self.device)
            data['done'] = T.as_tensor(data['done'], dtype=T.float32, device=self.device)
            data['weights'] = T.as_tensor(data['weights'], device=self.device)
            weights = data['weights']
            idxs = data['idxs']
            element_loss = self._compute_q_loss(data, self.n_step)
            loss = T.mean(weights * element_loss)
        else:
            data['state'] = {k:T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                for k,v in data['state'].items()}
            data['act'] = T.as_tensor(data['act'], dtype=T.int64, device=self.device)
            data['rew'] = T.as_tensor(data['rew'], dtype=T.float32, device=self.device)
            data['done'] = T.as_tensor(data['done'], dtype=T.float32, device=self.device)
            loss = self._compute_ep_loss(data)

        self.q_loss.append(loss.item()) # store loss 
        self.mac.optimizer.zero_grad()
        self.mixer.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.mac.parameters(), self.gradient_clip)
        clip_grad_norm_(self.mixer.parameters(), self.gradient_clip)
        self.mac.optimizer.step()
        self.mixer.optimizer.step()
    
        if update_step % self.update_target_step == 0:
            self._hard_update()

        if not self.use_rnn:
            loss_for_priority = element_loss.detach().cpu().numpy()
            new_priority = loss_for_priority + self.prior_eps
            return new_priority, idxs
        
        else:
            return loss

    def save_weights(self, mac_path, mixer_path):
        self.mac.save(mac_path)
        self.mixer.save(mixer_path)

    def load_weights(self, mac_path, mixer_path):
        self.mac.load(mac_path)
        self.mixer.load(mixer_path)

    def _hard_update(self):
        self.target_mac.load_state_dict(self.mac.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _soft_update(self, tau):
        with T.no_grad():
            for q1, q1_tar in zip(self.mac.parameters(), self.target_mac.parameters()):
                q1_tar.data = q1_tar.data.mul(tau)
                q1_tar.data = q1_tar.data.add((1-tau)*q1.data)
            for qmix, qmix_tar in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                qmix_tar.data = qmix_tar.data.mul(tau)
                qmix_tar.data = qmix_tar.data.add((1-tau)*qmix.data)

    def get_avg_loss(self):
        assert(len(self.q_loss)) > 0 
        critic_avg_loss = np.mean(self.q_loss)
        self.q_loss = []

        return dict(q_loss=critic_avg_loss)

            
    
