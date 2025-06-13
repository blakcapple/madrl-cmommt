import os
from utils.arguments import read_agrs
from pathlib import Path 
import os 
from eval.parallel_eval import test_with_single_params, test_with_various_params
import torch 

if __name__ == '__main__':
    args = read_agrs()
    torch.set_num_threads(1) # 防止torch多线程导致的性能下降
    # set config dir 
    if args.learning_stage == 2:
        config_dir =  os.path.join(os.path.dirname(__file__), 'configs/meta_policy')
    else:
        config_dir =  os.path.join(os.path.dirname(__file__), 'configs/default')

    # make run_dir 
    if args.full_test:
        run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'full_test' / args.algo
    else:
        run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'test' / args.algo

    if not run_dir.exists():
        os.makedirs(str(run_dir))
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = str(run_dir / curr_run)
    
    if args.full_test:
        test_with_various_params(args, run_dir, config_dir)
    else:
        test_with_single_params(args, run_dir, config_dir)




