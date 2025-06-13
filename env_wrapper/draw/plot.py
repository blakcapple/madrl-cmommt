import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

# smmoth average
def running_mean(x,n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('plot_rw', default=True, action='store_true')
    parser.add_argument('plot_loss', default=False, action='store_true')
    parser.add_argument('--window_size', default=100, type=int)
    args = parser.parse_args()

    for i , log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        train_pattern = r"episodes: (?P<ep>\d+), episode_len: (?P<ep_len>\d+),  episode reward: \[(?P<rw>.\d+.\d+)], mean_reward: (?P<mrw>\d+.\d+)"

        train_episode = []
        
        train_reward = []
        train_loss = []

        for r in re.findall(train_pattern, log):
            train_reward.append(r[2])
        train_reward = np.array(train_reward).astype(np.float)
        smooth_reward = running_mean(train_reward, 100)
        
    if args.plot_rw:
        fig, ax = plt.subplots()
        ax.plot(range(len(smooth_reward)), smooth_reward)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        plt.savefig('data/reward_plot.png')

if __name__ == '__main__':
    main()

