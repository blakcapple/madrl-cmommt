import json
import os
import sys
import glob
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from pathlib import Path
base_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_dir)
from matplotlib.path import Path
import matplotlib.patches as patches
from math import sin, cos, atan2, sqrt, pow
from env_wrapper.draw.geometry import get_2d_car_model, get_2d_uav_model
from env_wrapper.draw.vis_util import get_uav_model, get_car_model, draw_agent_3d, rgba2rgb
import seaborn as sns


plt_colors = [[0.8500, 0.3250, 0.0980], [0.0, 0.4470, 0.7410], [0.4660, 0.6740, 0.1880],
              [0.4940, 0.1840, 0.5560],
              [0.9290, 0.6940, 0.1250], [0.3010, 0.7450, 0.9330], [0.6350, 0.0780, 0.1840]]

abs_path = os.path.abspath('.')
plot_save_dir = abs_path + os.path.normpath('/')
# print(plot_save_dir)
base_fig_name = "{test_case}_{policy}_{num_agents}agents{step}.{extension}"


def draw_traj_3d(ax, agents_info, agents_traj_list, step_num_list, current_step):
    for idx, agent_traj in enumerate(agents_traj_list):
        info = agents_info[idx]
        group = info['gp']
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num:
            current_step = ag_step_num - 1

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        pos_z = np.ones_like(pos_y)
        alpha = agent_traj['alpha']
        beta = np.zeros_like(alpha)
        gamma = np.zeros_like(alpha)

        # 绘制实线
        # plt.plot(pos_x[:current_step], pos_y[:current_step], pos_z[:current_step],
        #          color=plt_color, ls='-', linewidth=2)
        # 绘制渐变线
        colors = np.zeros((current_step, 4))
        colors[:, :3] = plt_color
        colors[:, 3] = np.linspace(0.2, 1., current_step)
        colors = rgba2rgb(colors)
        alphas = np.linspace(0.0, 1.0, current_step + 1)
        for step in range(current_step):
            ax.scatter(pos_x[step], pos_y[step], pos_z[step], color=colors[step], s=3, alpha=alphas[step])
        # ax.scatter(pos_x[:current_step], pos_y[:current_step], pos_z[:current_step], color=colors, s=3, alpha=0.05)

        # # Also display circle at agent position at end of trajectory
        # ind = agent.step_num + last_index
        # alpha = 0.7
        # c = rgba2rgb(plt_color + [float(alpha)])
        # ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
        #                         radius=agent.radius, fc=c, ec=plt_color))
        # y_text_offset = 0.1
        # ax.text(agent.global_state_history[ind, 1] - 0.15,
        #         agent.global_state_history[ind, 2] + y_text_offset,
        #         '%d' % agent.id, color=plt_color)
        #####################################################################

        pos_global_frame = [pos_x[current_step], pos_y[current_step], pos_z[current_step]]
        heading_global_frame = [alpha[current_step], beta[current_step], gamma[current_step]]
        # print('pos_global_frame=', pos_global_frame)
        # print('heading_global_frame=', heading_global_frame)

        if group == 0:
            my_agent_model = get_uav_model()
        else:
            my_agent_model = get_car_model()
        draw_agent_3d(ax=ax,
                      pos_global_frame=pos_global_frame,
                      heading_global_frame=heading_global_frame,
                      my_agent_model=my_agent_model)


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
    patch = patches.PathPatch(path, facecolor='orange', lw=2)

    ax.add_patch(patch)

def draw_traj_2d0(ax, agents_info, agents_traj_list, step_num_list, current_step):
    for idx, agent_traj in enumerate(agents_traj_list):
        info = agents_info[idx]
        group = info['gp']
        radius = info['radius']
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num:
            current_step = ag_step_num - 1

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['alpha']

        # 绘制实线
        # plt.plot(pos_x[:current_step], pos_y[:current_step],
        #          color=plt_color, ls='-', linewidth=2)
        # 绘制渐变线
        colors = np.zeros((current_step, 4))
        colors[:, :3] = plt_color
        colors[:, 3] = np.linspace(0.2, 1., current_step)
        colors = rgba2rgb(colors)
        alphas = np.linspace(0.0, 1.0, current_step + 1)
        for step in range(current_step):
            ax.scatter(pos_x[step], pos_y[step], color=colors[step], s=3, alpha=alphas[step])
        # ax.scatter(pos_x[:current_step], pos_y[:current_step], color=colors, s=3, alpha=0.05)

        # # Also display circle at agent position at end of trajectory
        # ind = agent.step_num + last_index
        # alpha = 0.7
        # c = rgba2rgb(plt_color + [float(alpha)])
        # ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
        #                         radius=agent.radius, fc=c, ec=plt_color))
        # y_text_offset = 0.1
        # ax.text(agent.global_state_history[ind, 1] - 0.15,
        #         agent.global_state_history[ind, 2] + y_text_offset,
        #         '%d' % agent.id, color=plt_color)
        #####################################################################
        ax.add_patch(plt.Circle((pos_x[current_step], pos_y[current_step]), radius=radius / 0.10,
                                fc=plt_color, ec=plt_color))


def draw_traj_2d(ax, agents_info, agents_traj_list, step_num_list, current_step, show=False):
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_id = agents_info[idx]['id']
        agent_gp = agents_info[idx]['gp']
        agent_rd = agents_info[idx]['radius']
        info     = agents_info[idx]
        group = info['gp']
        radius = info['radius']
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num-1:
            current_step = ag_step_num - 1

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['alpha']

        # 绘制实线
        # plt.plot(pos_x[:current_step], pos_y[:current_step],
        #          color=plt_color, ls='-', linewidth=2)
        # 绘制渐变线
        colors = np.zeros((current_step, 4))
        colors[:, :3] = plt_color
        colors[:, 3] = np.linspace(0.2, 1., current_step)
        colors = rgba2rgb(colors)
        alphas = np.linspace(0.0, 1.0, current_step + 1)
        ax.scatter(pos_x[:current_step], pos_y[:current_step], color=colors, s=3, alpha=0.5)
        if group == 0:
            my_model = get_2d_uav_model(size=agent_rd*6)
        else:
            my_model = get_2d_car_model(size=agent_rd*6)
        pos = [pos_x[current_step], pos_y[current_step]]
        heading = alpha[current_step]
        draw_agent_2d(ax, pos, heading, my_model)
    if show:
        plt.pause(0.00001)
        

def plot_episode_3d(agents_info, agents_traj_list, step_num_list, in_evaluate_mode=None, obstacles=None,
                    plot_save_dir=None, plot_policy_name=None, show_prob_matrix=False, config_info=None,
                    fig_size=(10, 8), show=False):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    # total_step = len(agents_traj_list[0])
    # print('num_agents:', num_agents, 'total_step:', total_step)
    if show_prob_matrix:
        matrix = np.load('log/matrix.npy')
        grid = config_info['Grid']
    else:
        matrix = None
        grid = None

    while current_step < total_step:
        fig = plt.figure(0)
        fig.set_size_inches(fig_size[0], fig_size[1])

        # ax = fig.add_subplot(1, 1, 1)
        ax = Axes3D(fig)

        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               xlim=(-5, 45),
               ylim=(-5, 45),
               zlim=(0, 5),
               # xticks=np.arange(-5, 45, 1),
               # yticks=np.arange(-5, 45, 1),
               # zticks=np.arange(0, 5, 1)
               )

        # ax.view_init(elev=15,  # 仰角
        #              azim=60  # 方位角
        #              )

        # plt.grid()
        s1 = time.time()
        if show_prob_matrix:
            draw_probability_matrix_3d(ax, matrix, grid, current_step)
        s2 = time.time()
        print('draw_prob', s2 - s1)

        draw_traj_3d(ax, agents_info, agents_traj_list, step_num_list, current_step)
        s3 = time.time()
        print('draw_traj', s3 - s2)
        if obstacles:
            pass

        fig_name = base_fig_name.format(
            policy="policy_name",
            num_agents=num_agents,
            test_case=str(0).zfill(3),
            step="_" + "{:05}".format(current_step),
            extension='png')
        filename = plot_save_dir + fig_name
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()
        # plt.savefig(filename, bbox_inches="tight")
        s4 = time.time()
        print('save', s4 - s3)
        print(filename)
        current_step += 5

        if show:
            plt.pause(0.0001)


def plot_episode_2d(agents_info, agents_traj_list, step_num_list, in_evaluate_mode=None, obstacles=None,
                    plot_save_dir=None, plot_policy_name=None, show_prob_matrix=False, config_info=None,
                    fig_size=(10, 8), show=False, matrix_file = 'log/matrix.npy', local_matrix_file='log/local_matrix_0.npy'):
    if show: plt.ion() # 打开交互式运行模式
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    ENV_BOUND = config_info['ENV_RANGE']
    if show_prob_matrix:
        matrix = np.load(matrix_file)
        grid = config_info['Grid']
    else:
        matrix = None
        grid   = None
    # local local matrix
    agent_num = config_info['Agent_Num']
    local_matrix_list = []
    local_agent_matrix_list = []
    local_merge_matrix_list = []
    for i in range(agent_num):
        local_matrix = np.load(f'log/local_matrix_{i}.npy')
        local_matrix_list.append(local_matrix)
        local_merge_matrix = np.load(f'log/local_merge_matrix_{i}.npy')
        local_merge_matrix_list.append(local_merge_matrix)
        local_agent_matrix = np.load(f'log/local_agent_matrix_{i}.npy')
        local_agent_matrix_list.append(local_agent_matrix)
    fig = plt.figure(0)
    fig.set_size_inches(fig_size[0], fig_size[1])
    fig_1 = plt.figure(1)
    fig_1.set_size_inches(fig_size[0], fig_size[1])
    fig_2 = plt.figure(2)
    fig_2.set_size_inches(fig_size[0], fig_size[1])
    fig_3 = plt.figure(3)
    fig_3.set_size_inches(fig_size[0], fig_size[1])
    while current_step < total_step:
        ax = fig.add_subplot(1, 1, 1)
        ax_1 = []
        ax_2 = []
        ax_3 = []
        for i in range(agent_num):
            sub_ax = fig_1.add_subplot(2,2,i+1)
            ax_1.append(sub_ax)
            sub_ax = fig_2.add_subplot(2,2,i+1)
            ax_2.append(sub_ax)
            sub_ax = fig_3.add_subplot(2,2,i+1)
            ax_3.append(sub_ax)
        ax.set(xlabel='X',
               ylabel='Y',
               xlim=(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1),
               ylim=(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1),
               xticks=np.arange(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1, 1),
               yticks=np.arange(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1, 1),
               )
        for i in range(agent_num):
            ax_1[i].set(
               xlabel='X',
               ylabel='Y',
               xlim=(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1),
               ylim=(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1),
               xticks=np.arange(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1, 1),
               yticks=np.arange(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1, 1),)
            ax_2[i].set(
               xlabel='X',
               ylabel='Y',
               xlim=(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1),
               ylim=(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1),
               xticks=np.arange(ENV_BOUND[0][0]-1, ENV_BOUND[0][1]+1, 1),
               yticks=np.arange(ENV_BOUND[1][0]-1, ENV_BOUND[1][1]+1, 1),)
            # ax_3[i].set(xlim=(0,7), ylim=(0,7))

        s1 = time.time()

        if show_prob_matrix:
            draw_probability_matrix_2d(ax, matrix, grid, current_step)
        for i in range(agent_num):
            draw_probability_matrix_2d(ax_1[i], local_matrix_list[i], grid, current_step)
            draw_probability_matrix_2d(ax_2[i], local_merge_matrix_list[i], grid, current_step)
            from scipy.ndimage import rotate
            # 对矩阵逆时针旋转90度后可视化到xy平面
            sns.heatmap(rotate(local_agent_matrix_list[i][current_step], 90), center=0.5, vmax=1, vmin=0, cmap='Reds', ax=ax_3[i])
        s2 = time.time()

        draw_traj_2d(ax, agents_info, agents_traj_list, step_num_list, current_step, show)
        s3 = time.time()

        if obstacles: pass

        fig_name = base_fig_name.format(
            policy="policy_name",
            num_agents=num_agents,
            test_case=str(0).zfill(3),
            step="_" + "{:05}".format(current_step),
            extension='png')
        filename = plot_save_dir + fig_name
        if not show: fig.savefig(filename) # save the figure
        breakpoint()
        fig.clf()  # close the entire current figure
        fig_1.clf()
        fig_2.clf()
        s4 = time.time()
        print('draw_prob', s2 - s1,   'draw_traj', s3 - s2, 'save', s4 - s3)
        print(filename)
        current_step += 1


def get_agent_traj(info_file, traj_file):
    with open(info_file, "r") as f:
        json_info = json.load(f)

    traj_list = []
    step_num_list = []
    agents_info = json_info['all_agent_info']
    obstacles_info = json_info['all_obstacle']
    config_info = json_info['some_config_info']
    for agent_info in agents_info:
        agent_id = agent_info['id']
        agent_gp = agent_info['gp']
        agent_rd = agent_info['radius']

        df = pd.read_excel(traj_file, index_col=0, sheet_name='agent' + str(agent_id))
        step_num_list.append(df.shape[0])
        traj_list.append(df.to_dict('list'))
        # for indexs in df.index:
        #     traj_info.append(df.loc[indexs].values[:].tolist())
        # traj_info = np.array(traj_info)

    return agents_info, traj_list, step_num_list, config_info


def png_to_gif(agent_num=None, plot_save_dir=plot_save_dir):
    fig_name = base_fig_name.format(
        policy="policy_name",
        num_agents=agent_num,
        test_case=str(0).zfill(3),
        step="_*",
        extension='png')
    last_fig_name = base_fig_name.format(
        policy="policy_name",
        num_agents=agent_num,
        test_case=str(0).zfill(3),
        step="",
        extension='png')
    all_filenames = plot_save_dir + fig_name
    print(all_filenames)
    last_filename = plot_save_dir + last_fig_name

    # Dump all those images into a gif (sorted by timestep)
    filenames = glob.glob(all_filenames)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)

    # Save the gif in a new animations sub-folder
    animation_filename = base_fig_name.format(
        policy="policy_name",
        num_agents=agent_num,
        test_case=str(0).zfill(3),
        step="",
        extension='gif')
    animation_save_dir = plot_save_dir + "animations/"
    os.makedirs(animation_save_dir, exist_ok=True)
    animation_filename = animation_save_dir + animation_filename
    imageio.mimsave(animation_filename, images, duration=0.1)

    # convert .gif to .mp4
    try:
        import moviepy.editor as mp
    except imageio.core.fetching.NeedDownloadError:
        imageio.plugins.ffmpeg.download()
        import moviepy.editor as mp
    clip = mp.VideoFileClip(animation_filename)
    clip.write_videofile(animation_filename[:-4] + ".mp4")


def draw_probability_matrix_3d(ax, matrix, grid, current_step):
    current_matrix = matrix[current_step]
    for x_index, prob_list in enumerate(current_matrix):
        x_area = [x_index * grid, (x_index + 1) * grid]
        for y_index, prob in enumerate(prob_list):
            y_area = [y_index * grid, (y_index + 1) * grid]
            x_list = [x_area[0], x_area[0], x_area[1], x_area[1]]
            y_list = [y_area[0], y_area[1], y_area[1], y_area[0]]
            z_list = [0, 0, 0, 0]
            pannel = [list(zip(x_list, y_list, z_list))]
            ax.add_collection3d(Poly3DCollection(pannel, facecolors=(1, 0, 0), alpha=prob))


def draw_probability_matrix_2d(ax, matrix, grid, current_step):
    current_matrix = matrix[current_step]
    for x_index, prob_list in enumerate(current_matrix):
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


draw_type_dict = {
    '2d': plot_episode_2d,
    '3d': plot_episode_3d
}

def gif_plot(info_file, traj_file, matrix_file, plot_save_dir, select_map_dim = '2d', show_prob_matrix=True, show=False):
    agents_info, traj_list, step_num_list, config_info = get_agent_traj(info_file, traj_file)
    draw_type_dict[select_map_dim](agents_info, traj_list, step_num_list, plot_save_dir=plot_save_dir, show=show,
                                show_prob_matrix=show_prob_matrix, config_info=config_info, matrix_file=matrix_file)
    png_to_gif(len(step_num_list), plot_save_dir=plot_save_dir)

if __name__ == '__main__':

    info_file = 'log/env_cfg.json'
    traj_file = "log/trajs.xlsx"
    select_map_dim = '2d'

    show_prob_matrix = True

    agents_info, traj_list, step_num_list, config_info = get_agent_traj(info_file, traj_file)
    draw_type_dict[select_map_dim](agents_info, traj_list, step_num_list, plot_save_dir=plot_save_dir, show=True,
                                show_prob_matrix=show_prob_matrix, config_info=config_info)

    png_to_gif(len(step_num_list))
