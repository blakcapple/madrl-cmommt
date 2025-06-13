import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

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
        train_pattern = r"episodes: (?P<ep>\d+), episode_len: (?P<ep_len>\d+), episode reward: (?P<rw>\d+.\d+)"
        train_episode = []
        
        train_reward = []
        train_loss = []

        for r in re.findall(train_pattern, log):
            train_reward.append(r[2])
        train_reward = np.array(train_reward).astype(np.float)
        smooth_reward = running_mean(train_reward, args.window_size)
        
    if args.plot_rw:
        fig, ax = plt.subplots()
        ax.plot(range(len(smooth_reward)), smooth_reward)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        plt.savefig('reward_plot.png')



def plot_rw(log_file, window_size=100, phase='train'):
    with open(log_file, 'r') as file:
        log = file.read()

    if phase == 'train':
        train_pattern = r"phase: TRAIN, episodes: (?P<ep>\d+), episode_len: (?P<ep_len>\d+), episode reward: (?P<rw>\d+.\d+)"
    else:
        train_pattern = r"phase: TEST, episodes: (?P<ep>\d+), episode_len: (?P<ep_len>\d+), episode reward: (?P<rw>\d+.\d+)"

    train_reward = []

    for r in re.findall(train_pattern, log):
        train_reward.append(r[2])
    train_reward = np.array(train_reward).astype(np.float)
    smooth_reward = running_mean(train_reward, window_size)
    
    fig, ax = plt.subplots()
    ax.plot(range(len(smooth_reward)), smooth_reward)
    ax.set_xlabel('Episode')
    ax.set_ylabel(phase.capitalize() + ' Reward')
    log_path = os.path.dirname(log_file)
    save_path = os.path.join(log_path, phase.capitalize() + 'Reward_plot.png')
    plt.savefig(save_path)
    plt.close()

def plot_qloss(log_file, window_size=100):
    with open(log_file, 'r') as file:
        log = file.read()
    loss_pattern = r'q_loss: (?P<rw>\d+.\d+)'
    loss = []
    for r in re.findall(loss_pattern, log):
        loss.append(r)
    loss = np.array(loss).astype(np.float)
    smooth_loss = running_mean(loss, window_size)

    fig, ax = plt.subplots()
    ax.plot(range(len(smooth_loss)), smooth_loss)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Averge QLoss')
    log_path = os.path.dirname(log_file)
    save_path = os.path.join(log_path,'QLoss.png')
    plt.savefig(save_path)
    plt.close()
    
def plot_piloss(log_file, window_size=100):
    with open(log_file, 'r') as file:
        log = file.read()
    loss_pattern = r'pi_loss: (?P<rw>\d+.\d+)'
    loss = []
    for r in re.findall(loss_pattern, log):
        loss.append(r)
    loss = np.array(loss).astype(np.float)
    smooth_loss = running_mean(loss, window_size)

    fig, ax = plt.subplots()
    ax.plot(range(len(smooth_loss)), smooth_loss)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Averge PiLoss')
    log_path = os.path.dirname(log_file)
    save_path = os.path.join(log_path,'PiLoss.png')
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()

