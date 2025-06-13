import os
from train.train_apex import train_apex 
from train.train_dqn import train_dqn 
from train.train_ppo import train_mappo 
from utils.arguments import read_agrs
from pathlib import Path 
import os 

if __name__ == '__main__':
    args = read_agrs()
    
    # set config dir 
    if args.learning_stage == 2:
        config_dir =  os.path.join(os.path.dirname(__file__), 'configs/meta_policy')
    else:
        config_dir =  os.path.join(os.path.dirname(__file__), 'configs/default')

    # make run_dir
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / args.algo / f'stage{args.learning_stage}'
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = str(run_dir / curr_run)
    
    if 'iql' in args.algo or 'qmix' in args.algo or 'vdn' in args.algo:
        train_dqn(args, run_dir, config_dir)
    elif 'apex' in args.algo:
        train_apex(args, run_dir, config_dir)
    elif 'ppo' in args.algo:
        train_mappo(args, run_dir, config_dir)
    else: 
        raise NotImplementedError


