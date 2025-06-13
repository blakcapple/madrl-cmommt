import json
import os
import sys
import glob
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
base_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_dir)
from matplotlib.path import Path
import matplotlib.patches as patches
from math import sin, cos, atan2, sqrt, pow
from env_wrapper.draw.geometry import get_2d_car_model, get_2d_uav_model
from env_wrapper.draw.vis_util import rgba2rgb
from matplotlib.patches import Circle
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

base_fig_name = "{pursuers_num}pursuer_{evaders_num}evader_{step}.{extension}"

def convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame
    for point in agent_model:
        x = point[0]
        y = point[1]
        # 进行航向计算
        l = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2  # 改加 - np.pi / 2 因为画模型的时候UAV朝向就是正北方向，所以要减去90°
        point[0] = l * cos(alpha_) + pos_global_frame[0]
        point[1] = l * sin(alpha_) + pos_global_frame[1]


def draw_agent_2d(ax, pos_global_frame, heading_global_frame, my_agent_model, color='blue'):
    agent_model = my_agent_model
    convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame)

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(agent_model, codes)

    # 第二步：创建一个patch，路径依然也是通过patch实现的，只不过叫做pathpatch
    patch = patches.PathPatch(path, facecolor=color, lw=1, edgecolor=color)

    ax.add_patch(patch)

def draw_traj_2d(ax, agents_info, agents_traj_list, step_num_list, current_step, show=False):
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_id = agents_info[idx]['id']
        agent_gp = agents_info[idx]['gp']
        agent_rd = agents_info[idx]['radius']
        sec_radius = agents_info[idx]['sec_radius']
        comm_radius = agents_info[idx]['commu_radius']
        info     = agents_info[idx]
        group = info['gp']
        radius = info['radius']
        if agent_gp == 0:
            plt_color = [0,0,1]
        elif agent_gp == 1:
            plt_color = [0, 0, 0]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num-1:
            current_step = ag_step_num - 1

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['alpha']
        if agent_gp == 0:
            cir1 = Circle(xy = (pos_x[current_step], pos_y[current_step]), radius=sec_radius, alpha=0.1)
            cir2 = Circle(xy = (pos_x[current_step], pos_y[current_step]), radius=comm_radius, alpha=0.05)
            txt = ax.text(pos_x[current_step]+0.1, pos_y[current_step]+0.1, str(agent_id), fontsize=10)
            ax.add_patch(cir1)
            ax.add_patch(cir2)
        # 绘制渐变线（保留最近200步的轨迹）
        colors = np.zeros((min(current_step, 200), 4))
        colors[:, :3] = plt_color
        colors[:, 3] = np.linspace(0.1, 1., min(current_step, 200), endpoint=True)
        colors = rgba2rgb(colors)
        ax.scatter(pos_x[max(current_step-200, 0):current_step],  # 绘制轨迹
                    pos_y[max(current_step-200, 0):current_step], 
                    color=colors, s=3, alpha=0.5)
        if agent_gp == 0:
            my_model = get_2d_uav_model(size=agent_rd*10)
            color = 'blue'
        else:
            my_model = get_2d_car_model(size=agent_rd*10)
            color = 'black'
        pos = [pos_x[current_step], pos_y[current_step]]
        heading = alpha[current_step]
        draw_agent_2d(ax, pos, heading, my_model, color)
    if show:
        plt.pause(0.00001)


def draw_local_matrix(local_agent_matrix_path):
    from matplotlib.ticker import MaxNLocator
    plt.ion()
    fig = plt.figure(figsize=(10,8))
    local_agent_matrix = np.load(local_agent_matrix_path, allow_pickle=True)
    n = local_agent_matrix.shape[1]
    print(local_agent_matrix.shape)
    steps = local_agent_matrix.shape[0]
    row = (n + 3) // 4
    # for i in range(n):
    #     ax = fig.add_subplot(1, n, i + 1)
    for s in range(steps):
        for i in range(n):
            ax = fig.add_subplot(row, 4, i + 1, aspect='equal')
            draw_probability_matrix_2d(ax, local_agent_matrix[s][i], grid=0.5)
            ax.set_title('agent' + str(i))
            ax.set(xlim=(0,5), ylim=(0,5))
            # 设置坐标轴刻度间隔为 1
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.pause(0.1)
        fig.clf() 


def plot_episode_2d(agents_info, agents_traj_list, step_num_list, config_info, plot_save_dir, 
                    show_prob_matrix=False, fig_size=(10, 8), online=False, matrix_file = 'log/matrix.npy', local_matrix_file=None, speedup_factor=5, draw_local_matrix=False):
    os.makedirs(plot_save_dir, exist_ok=True)
    if online: plt.ion() # 打开交互式运行模式
    current_step = 0
    total_step = max(step_num_list)
    ENV_BOUND = config_info['env_range']
    if show_prob_matrix:
        matrix = np.load(matrix_file)
        local_agent_matrix = np.load(local_matrix_file, allow_pickle=True)
        grid = config_info['grid']
        
    fig = plt.figure(figsize=fig_size)
    while current_step < total_step:
        if draw_local_matrix:
            gs = gridspec.GridSpec(2, 6)
            ax = fig.add_subplot(gs[:, :2], aspect="equal")
        else:
            ax = fig.add_subplot()
        # 设置坐标轴属性
        ax.set(xlabel='X',
            ylabel='Y',
            xlim=(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1),
            ylim=(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1),
            xticks=np.arange(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1, 1),
            yticks=np.arange(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1, 1),
            )
        if show_prob_matrix:
            if draw_local_matrix:
                row = (local_agent_matrix.shape[1] + 3) // 4
                col = 4
                for i in range(row):
                    for j in range(col):
                        sub_ax = fig.add_subplot(gs[i, j+2], aspect="equal")
                        draw_probability_matrix_2d(sub_ax, local_agent_matrix[current_step][i*4+j], grid=grid)
                        sub_ax.set_title('agent' + str(i*4+j), fontsize=8)
                        sub_ax.set(xlim=(0,20), ylim=(0,20))
                        sub_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        sub_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                        sub_ax.tick_params(axis='both', which='major', labelsize=6)
            draw_probability_matrix_2d(ax, matrix[current_step], grid)
        draw_traj_2d(ax, agents_info, agents_traj_list, step_num_list, current_step, online)
        # rect = patches.Rectangle((7.75, 7.75), 4+0.5, 4+0.5, linewidth=1, edgecolor='black', linestyle='--', facecolor='none')
        # ax.add_patch(rect)
        # plt.axis('off')  # 隐藏所有坐标轴和边框
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig_name = base_fig_name.format(
        evaders_num=config_info['evader_num'],
        pursuers_num=config_info['pursuer_num'],
        step="{:05}".format(current_step),
        extension='png')
        filename = os.path.join(plot_save_dir,fig_name)
        fig.savefig(filename, bbox_inches='tight')
        print(filename)
        fig.clf() 
        increase_step = speedup_factor ## 渲染加速比例
        current_step += increase_step

def get_agent_traj(info_file, traj_file):
    with open(info_file, "r") as f:
        json_info = json.load(f)

    traj_list = []
    step_num_list = []
    agents_info = json_info['all_agent_info']
    config_info = json_info['some_config_info']
    for agent_info in agents_info:
        agent_id = agent_info['id']
        df = pd.read_excel(traj_file, index_col=0, sheet_name='agent' + str(agent_id))
        step_num_list.append(df.shape[0])
        traj_list.append(df.to_dict('list'))

    return agents_info, traj_list, step_num_list, config_info


def png_to_gif(plot_save_dir, config_info):
    fig_name = base_fig_name.format(
        pursuers_num = config_info['pursuer_num'],
        evaders_num = config_info['evader_num'],
        step="*",
        extension='png')
    all_filenames = os.path.join(plot_save_dir, fig_name)
    print(all_filenames)

    # Dump all those images into a gif (sorted by timestep)
    filenames = glob.glob(all_filenames)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        # os.remove(filename)
    animation_save_dir = plot_save_dir
    os.makedirs(animation_save_dir, exist_ok=True)
    animation_filename = os.path.join(animation_save_dir, '8pursuer_8evader.gif')
    imageio.mimsave(animation_filename, images, duration=0.1)

def save_video(image_folder, video_name, config_info):
    import cv2
    # 视频编解码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[1]))
    height, width, _ = img.shape
    # 创建视频写入器
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
    fig_name = base_fig_name.format(
        pursuers_num = config_info['pursuer_num'],
        evaders_num = config_info['evader_num'],
        step="*",
        extension='png')
    img_path = os.path.join(image_folder, fig_name)
    filenames = glob.glob(img_path)
    filenames.sort()
    # 遍历所有的图片，将其添加到视频中
    for filename in filenames:
        img = cv2.imread(filename)
        video.write(img)
        # os.remove(filename)
    # 释放视频写入器资源
    video.release()
    
def draw_probability_matrix_2d(ax, matrix, grid):
    for x_index, prob_list in enumerate(matrix):
        for y_index, prob in enumerate(prob_list):
            ax.add_patch(
                plt.Rectangle(
                    (x_index * grid, y_index * grid),  # (x,y)矩形左下角
                    grid,  # width长
                    grid,  # height宽
                    color=(1, 0, 0),
                    alpha=prob
                )
            )

def gif_plot(info_file, traj_file, matrix_file, local_matrix_file, plot_save_dir, show_prob_matrix=True, online=False, speedup_factor=5):
    agents_info, traj_list, step_num_list, config_info = get_agent_traj(info_file, traj_file)
    draw_local_matrix = False
    if not draw_local_matrix:
        fig_size = (10,8)
    else:
        fig_size = (16, 5)
    plot_episode_2d(agents_info, traj_list, step_num_list, config_info, plot_save_dir,show_prob_matrix, fig_size, online, matrix_file, local_matrix_file, speedup_factor, draw_local_matrix)
    video_path = os.path.join(plot_save_dir, 'animation.avi')
    save_video(plot_save_dir, video_path, config_info)
    png_to_gif(str(plot_save_dir), config_info)


def plot_matrix(matrix_file, step_num_list, config_info, fig_size, plot_save_dir):
    current_step = 0
    total_step = max(step_num_list)
    matrix = np.load(matrix_file, allow_pickle=True)
    grid = config_info['grid']
    fig = plt.figure(figsize=fig_size)
    num = matrix.shape[1]
    ax = fig.add_subplot()
    while current_step < total_step:
        for i in range(num):
            draw_probability_matrix_2d(ax, matrix[current_step][i], grid)
            # ax.set_title('agent' + str(i), fontsize=8)
            ax.set(xlim=(0,20), ylim=(0,20))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=6)
            fig_name = f'agent_{i}_step{current_step}'
            filename = os.path.join(plot_save_dir,fig_name)
            plt.axis('off')  # 隐藏所有坐标轴和边框
            fig.savefig(filename, bbox_inches='tight')
            ax.cla()
        current_step += 10

def plot_agent_matrix(matrix_file, step_num_list, config_info, fig_size, plot_save_dir):
    current_step = 0
    total_step = max(step_num_list)
    matrix = np.load(matrix_file, allow_pickle=True)
    grid = config_info['grid']
    fig = plt.figure(figsize=fig_size)
    num = matrix.shape[1]
    ax = fig.add_subplot()
    while current_step < total_step:
        for i in range(num):
            draw_probability_matrix_2d(ax, matrix[current_step][i], 0.5)
            # ax.set_title('agent' + str(i), fontsize=8)
            ax.set(xlim=(0,5), ylim=(0,5))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=6)
            fig_name = f'agent_{i}_step{current_step}'
            filename = os.path.join(plot_save_dir,fig_name)
            plt.axis('off')  # 隐藏所有坐标轴和边框
            fig.savefig(filename)
            ax.cla()
        current_step += 10

if __name__ == '__main__':
    import pathlib
    info_file = pathlib.Path(__file__).parent / 'log' / 'env_cfg.json'
    traj_file = pathlib.Path(__file__).parent / 'log' / 'trajs.xlsx'
    matrix_file = pathlib.Path(__file__).parent / 'log' / 'matrix.npy'    
    local_matrix_file_list = []
    local_matrix_file = str(pathlib.Path(__file__).parent / 'log' / f'local_matrix.npy')
    plot_save_dir = os.path.join(pathlib.Path(__file__).parent, 'animations')
    # gif_plot(info_file, traj_file, matrix_file, local_matrix_file, plot_save_dir, show_prob_matrix=True, online=False, speedup_factor=10)
    matrix_file = pathlib.Path(__file__).parent / 'log' / 'local_matrix.npy'    
    agents_info, traj_list, step_num_list, config_info = get_agent_traj(info_file, traj_file)
    # plot_agent_matrix(matrix_file, step_num_list, config_info, (10,8), plot_save_dir)
    plot_matrix(matrix_file, step_num_list, config_info, (10,8), plot_save_dir)





