import torch as T
import numpy as np 
from algos.utils.epsilon_decay import DecayThenFlatSchedule
from gym import spaces
from .rl_agent import RLAgent
class DQNAgent(RLAgent):

    def __init__(self, env_info, all_config, device, policy):
        super().__init__()
        self.cfg = all_config
        self.per = self.cfg.per
        self.action_shape = env_info.action_shape
        self.action_space = env_info.action_space
        self.num_agents = env_info.agent_num
        eps_max = self.cfg.eps_max
        eps_min = self.cfg.eps_min
        decay_frame = self.cfg.decay_frame
        decay_rule = self.cfg.decay_rule
        self.use_rnn = self.cfg.use_rnn
        self.epsilon_schedule = DecayThenFlatSchedule(eps_max, eps_min, decay_frame, decay=decay_rule)

        self.policy = policy
        self.device = device
        
    def random_sample(self, action_mask=None):
        
        if action_mask is not None:
            action_probability = []
            actions = []
            for i in range(self.num_agents):
                action_probability = T.ones(self.action_space.n)
                action_probability[action_mask[i]] = 0
                action = T.multinomial(action_probability, num_samples=1)
                actions.append(action.item())
        else:
            actions = [self.action_space.sample() for i in range(self.num_agents)] 
        actions = np.array(actions).reshape(-1, self.action_shape)
        return actions
    
    def select_actions(self, ob, deterministic=False, env_step=None, hidden_state=None):
        
        state = {k:T.as_tensor(v[None,:], dtype=T.float32, device=self.device) for k,v in ob.items()}
        
        if self.use_rnn:
            action_value, hidden_state = self.policy.mac(state, hidden_state)
        else:
            action_value = self.policy.mac(state)
        q_value = np.mean(np.max(action_value.cpu().detach().numpy(), axis=1))
        if 'action_mask' in state.keys():
            action_mask = state['action_mask'].bool().squeeze(0)
            action_value[action_mask] = -np.inf
        if deterministic:
            actions = T.argmax(action_value, dim=1).tolist()
        else:
            self.epsilon = self.epsilon_schedule.eval(env_step)
            if np.random.random() > self.epsilon:
                actions = T.argmax(action_value, dim=1).tolist()
            else:
                actions = self.random_sample(action_mask if 'action_mask' in state.keys() else None)
        actions = np.array(actions).reshape(-1, self.action_shape)
        if deterministic:
            return actions
        if self.use_rnn:
            return actions, q_value, hidden_state
        else:
            return actions, q_value
    
    def train(self, data, train_step):
        if self.per:
            priority, idxs = self.policy.learn(data, train_step)
            return priority, idxs
        else:
            self.policy.learn(data, train_step)
    
    def save_weight(self, mac_path, mixer_path=None):
        if hasattr(self.policy, 'mixer'):
            self.policy.save_weights(mac_path, mixer_path)
        else:
            self.policy.save_weights(mac_path)

    def load_weight(self, mac_path, mixer_path=None):
        if hasattr(self.policy, 'mixer'): 
            self.policy.load_weights(mac_path, mixer_path)
        else:
            self.policy.load_weights(mac_path)




