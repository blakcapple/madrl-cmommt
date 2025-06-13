import numpy as np 

class BaseConfig:

    def __init__(self):
        
        # 环境中追逐者和逃避者的数量范围
        self.max_pursuer_num = 8
        self.min_pursuer_num = 8 
        self.max_evader_num = 8
        self.min_evader_num = 8
        self.dt = 0.1 # 单步时间长度
        self.total_steps = 3000  # 环境总步数
        self.pursuer_speed = 0.3 # 追逐者移动速度
        self.evader_speed = 0.1 # 逃避者移动速度
        self.search_radius = 1.25 # 追逐者搜索半径
        self.comm_radius = 2.5 # 追逐者通信半径
        self.evader_sec_radius = 1.5 # 逃避者感知半径
        self.max_angular_v = np.pi/6 # 追逐者最大角速度
        self.env_range = [[0, 20], [0, 20]] # 环境范围
        self.NEAR_GOAL_THRESHOLD = 0.5 # 用于逃避者逃避策略中的一个参数
        self.CAPTURE_RADIUS = 0.3 # not used
        self.pursuer_radius = 0.01 # 追逐者追逐半径大小
        self.evader_radius = 0.01 # 逃避者半径大小
        self.evader_policy = 'escape' # or random 逃避策略
        self.pursuer_policy = 'rl' # 追逐策略
        self.test = False # 是否是测试环境
        self.local_grid = 0.5 # 用于计算平均探索频率时的局部网格大小

        self.animation_colums = ['pos_x', 'pos_y', 'alpha'] # 用于可视化展示的信息
        
env_config = BaseConfig()