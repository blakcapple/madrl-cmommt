from .actor import PPOActor
from .critic import PPOCritic, PPOCentrialCritic
import torch
import numpy as np 
from algos.utils.epsilon_decay import DecayThenFlatSchedule
class PPOAC:
    
    def __init__(self, config, env_info, device):
        
        self.config = config
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.opti_eps = config['opti_eps']
        self.weight_decay = config['weight_decay']
        self.actor = PPOActor(config, env_info, device)
        self.critic = PPOCritic(config, env_info, device) if not config.use_central_critic else PPOCentrialCritic(config, env_info, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    
    def get_actions(self, obs, deterministic=False, env_step=None):

        actions, action_log_probs = self.actor(obs, deterministic)
        values = self.critic(obs)

        return values, actions, action_log_probs

    def get_values(self, obs):

        values = self.critic(obs)
        
        return values   

    def evaluate_actions(self, obs, action, active_masks_batch):

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, action, active_masks_batch)
        values = self.critic(obs)
        return values, action_log_probs, dist_entropy

    def act(self, obs, deterministic=False):

        actions, _ = self.actor(obs, deterministic)
        
        return actions

    def save(self, pth):

        torch.save(self.actor.state_dict(), pth+'/actor.pth')
        torch.save(self.critic.state_dict(), pth+'/critic.pth')
        
    def load_actor(self, pth):
        
        self.actor.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    




