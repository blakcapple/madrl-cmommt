import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from algos.buffer.ppo_shared_buffer import SharedDictReplayBuffer
from algos.policy.mappo import MAPPOTrainer as TrainAlgo
def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config, train_env, policy, eval_env=None):
        
        self.all_args = config
        self.envs = train_env
        self.eval_envs = eval_env
        self.device = config['device']
        self.env_info = config['env_info']
        self.num_agents = self.env_info.agent_num

        # parameters
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.eval_episode = self.all_args.eval_episode

        # dir
        self.load_actor = False 
        self.load_critic = False


        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir + '/logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir + '/models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # policy network
        self.policy = policy

        self.restore(self.load_actor, self.load_critic)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedDictReplayBuffer(self.all_args,self.n_rollout_threads,self.num_agents,\
                                        self.envs.observation_space,
                                        self.envs.action_space)
        
        self.logger = None

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        last_obs = {}
        for key, val in self.buffer.obs.items():
            last_obs[key] = torch.from_numpy(np.concatenate(val[-1])).to(device=self.device, dtype=torch.float32)
        next_values = self.trainer.policy.get_values(last_obs)
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, epoch, save_best=False):
        """Save policy's actor and critic networks."""
        if save_best:
            # remove last best model
            files = os.listdir(self.save_dir)
            for file in files:
                if 'best' in file:
                    file_path = os.path.join(self.save_dir, file)
                    os.remove(file_path)
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{epoch}_best.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{epoch}_best.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{epoch}.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{epoch}.pt")

    def restore(self, load_actor, load_critic):
        """Restore policy's networks from a saved model."""
        from pathlib import Path
        model_dir = Path(os.path.dirname(__file__)).resolve().parent / 'load_model'
        if load_actor:
            policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
        if load_critic:
            policy_critic_state_dict = torch.load(str(model_dir)+'/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
