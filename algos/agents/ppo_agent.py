from .rl_agent import RLAgent
from network.ac.ac import PPOAC
import torch
class PPOAgent(RLAgent):
    
    def __init__(self, policy:PPOAC, device):
        
        super().__init__()
        
        self.policy = policy
        self.device = device
        
    def select_actions(self, obs, deterministic):
        
        state = {k:torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in obs.items()}
        actions = self.policy.act(state, deterministic)
        return actions.detach().cpu().numpy()
    
    def load_weight(self, pth):
        
        self.policy.load_actor(pth)
