import numpy as np 
from copy import deepcopy

def create_matrix(grid, env_range):
    '''
    构建概率矩阵
    '''
    max_x = env_range[0][1]
    max_y = env_range[1][1]

    x = np.mgrid[grid:max_x:grid].tolist()
    y = np.mgrid[grid:max_y:grid].tolist()

    probability_matrix = []

    if x[-1] < max_x:
        x.append(np.array(max_x + grid / 2))
    if y[-1] < max_y:
        y.append(max_y + grid / 2)

    for x_ in x:
        per_x_list1 = []
        for y_ in y:
            per_x_list1.append(0.5)
        probability_matrix.append(per_x_list1)

    return np.array(probability_matrix)

def update_probability_matrix(env_range, gamma, matrix_grid, observed_pursuer, observed_evader, probability_matrix, search_radius=1.25, method='geometry'):
    '''
    更新概率矩阵
    '''
    # 向概率为0.5衰变
    probability_matrix[probability_matrix<0.5] = 1 - (1 - probability_matrix[probability_matrix<0.5]) * gamma
    probability_matrix[probability_matrix>0.5] = probability_matrix[probability_matrix>0.5] * gamma
    # 采样估计
    if method == 'sample':
        sample_num = 3
        radius_list = np.linspace(0.1, search_radius, sample_num, endpoint=True)
        for ag in observed_pursuer:
            pos_x = ag.pos[0]
            pos_y = ag.pos[1]
            for _, radius in enumerate(radius_list):  # 在依次衰减的搜索半径中采样
                theta_list = np.random.uniform(-np.pi, np.pi, max(int(10*radius/search_radius),2))
                for theta in theta_list:
                    x = pos_x + radius * np.cos(theta)
                    x = np.clip(x, 0, env_range)
                    y = pos_y + radius * np.sin(theta)
                    y = np.clip(y, 0, env_range)
                    pos = [x,y]
                    index_x = int(pos[0] // matrix_grid)-1 if pos[0] % matrix_grid == 0 and not pos[0] == 0 else int(pos[0] // matrix_grid)
                    index_y = int(pos[1] // matrix_grid)-1 if pos[1] % matrix_grid == 0 and not pos[1] == 0 else int(pos[1] // matrix_grid)
                    p = probability_matrix[index_x][index_y] - 0.1
                    p = max(0,min(p, 1))
                    probability_matrix[index_x][index_y] = p
    # 几何计算估计
    elif method == 'geometry':
        ratio = 0.13 # 渐变速率
        matrix_shape = probability_matrix.shape
        max_square = pow(matrix_grid, 2)
        record = set()
        prob_record = {}
        for ag in observed_pursuer:
            boudary_list = np.zeros((4,2))
            index_list = np.zeros((4,2))
            matrix = np.dot(np.array([[-1,1], [1,1], [-1,-1], [1,-1]]), 
                            np.array([[matrix_grid/2, 0],[0, matrix_grid/2]]))                
            for i in range(4):
                boudary_list[i][0] = ag.pos[0] + matrix[i, 0]
                boudary_list[i][0] = np.clip(boudary_list[i][0], 0, env_range)
                index_list[i][0] = int(boudary_list[i][0] // matrix_grid)
                boudary_list[i][1] = ag.pos[1] + matrix[i, 1]
                boudary_list[i][1] = np.clip(boudary_list[i][1], 0, env_range)
                index_list[i][1] = int(boudary_list[i][1] // matrix_grid)
            grid_px, grid_py = index_list[1][0]*matrix_grid, index_list[1][1]*matrix_grid

            for i in range(4):
                point = boudary_list[i]
                x_index = int(index_list[i][0])
                y_index = int(index_list[i][1])
                if point[0] == 0 or point[1] == 0 or point[0] == env_range or point[1] == env_range:
                    continue
                square = abs(point[0]-grid_px)*abs(point[1]-grid_py)
                dx = 1 if(point[0]-grid_px) > 0 else -1
                dy = 1 if (point[1] - grid_py) > 0 else -1
                if (x_index, y_index) not in record:
                    record.add((x_index, y_index))
                    prob_record[matrix_shape[0]*x_index+y_index] = [[square, dx, dy]]
                else:
                    prob_record[matrix_shape[0]*x_index+y_index].append([square, dx, dy])
        for k,v in prob_record.items():
            x_index = k // matrix_shape[0]
            y_index = k % matrix_shape[0] 
            direction = set()
            for i, info in enumerate(v):
                square, dx, dy = info[0], info[1], info[2]
                if i == 0:
                    p = square/max_square*0.5
                    probability_matrix[x_index][y_index] -= p*ratio
                    if probability_matrix[x_index][y_index] < 0:
                        probability_matrix[x_index][y_index] = 0
                    direction.add((dx, dy))
                else:
                    if (dx,dy) not in direction:
                        p = square/max_square*0.5
                        probability_matrix[x_index][y_index] -= p*ratio
                        probability_matrix[x_index][y_index] = max(0, probability_matrix[x_index][y_index])
                        direction.add((dx, dy))
                    else:
                        p = square/max_square*0.5 / 2
                        probability_matrix[x_index][y_index] -= p*ratio
                        if probability_matrix[x_index][y_index] < 0:
                            probability_matrix[x_index][y_index] = 0
    else:
        raise NotImplementedError  
    matrix_num = int(env_range // matrix_grid)
    for ag in observed_evader:
        evader_position = ag.pos
        evader_x_index = int(evader_position[0] // matrix_grid)
        evader_y_index = int(evader_position[1] // matrix_grid)
        evader_x_index = min(evader_x_index, matrix_num-1)
        evader_y_index = min(evader_y_index, matrix_num-1)
        probability_matrix[evader_x_index][evader_y_index] = 1.0

    return probability_matrix
    
def merge_local_probability_matrix(obs_dict, merge_before_matrix):

    """
    融合局部概率矩阵
    """
    merge_after_matrix = deepcopy(merge_before_matrix)
    for index in merge_before_matrix.keys():
        if index >= len(obs_dict['pursuer_observe_evader_dict']):
            break
        local_matrix = merge_before_matrix[index]
        for friend_id in obs_dict['friend_observe_dict'][index]:
            local_matrix = np.where(np.abs(local_matrix-0.5)>np.abs(merge_before_matrix[friend_id]-0.5), local_matrix, merge_before_matrix[friend_id])
        merge_after_matrix[index] = local_matrix

    return merge_after_matrix