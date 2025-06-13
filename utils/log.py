import logging
import os
from datetime import datetime


def init_log(save_path, pid=0):
    	
    # create unique directories
	# now = datetime.now().strftime("%b-%d_%H-%M-%S")  
	# save_path = os.path.join(path, now)
	os.makedirs(save_path, exist_ok=True)
	log_file = os.path.join(save_path, 'out.log')
	# create logger
	logger = logging.getLogger(f'sur_{pid}')
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler(log_file, mode='w')
	fh.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s, %(levelname)s: %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)
	return logger, save_path, log_file

def log_info(logger, cfg, env, args):
	logger.info('Train')		
	logger.info('ALGO INFO')
	logger.info('algo: {}'.format(args.algo))
	logger.info('share_reward:{}'.format(cfg['share_reward']))
	logger.info('collision_avoidance:{}'.format(env.collision_avoidance))
	logger.info('use_matrix:{}'.format(env.use_matrix))
	logger.info('ENV INFO')
	logger.info('env_range:{}'.format(env.env_range))
	logger.info('matrix_gamma:{}'.format(env.matrix_gamma))
	logger.info('dt:{}'.format(env.dt))
	logger.info('AGENT INFO')
	logger.info("pursuer_speed:{}, evader_speed:{}".format(env.pursuer_speed, env.evader_speed))
	logger.info('search_radius:{}'.format(env.search_radius))
	logger.info('evader_sec_radius:{}'.format(env.evader_sec_radius))
	logger.info('evader_policy:{}'.format(env.evader_policy))
	logger.info('communicate_radius:{}'.format(env.comm_radius))
	logger.info('ACTION INFO')
	logger.info('action: angular velocity')
	logger.info('action_dim:{}'.format(env.action_dim))
	if env.action_type == 'continue':
		logger.info('Use Continuous Action Space')
		logger.info("Action_Bound: {:.2f}".format(env.action_space.high))
	else:
		logger.info('Use Discrete Action Space')
		logger.info("Action_Bound: {:.2f}".format(env.max_action))
		logger.info("Action_Num: {:.2f}".format(env.action_space.n))

def log_test_info(logger, cfg, env):
	logger.info('Test')	
	logger.info('learning_stage:{}'.format(env.learning_stage))
	logger.info('collision_avoidance:{}'.format(env.collision_avoidance))
	logger.info('use_matrix:{}'.format(env.use_matrix))
	logger.info('ENV INFO')
	logger.info('env_range:{}'.format(env.env_range))
	logger.info('env_gamma:{}'.format(env.matrix_gamma))
	logger.info('dt:{}'.format(env.dt))
	logger.info('AGENT INFO')
	# logger.info("pursuer_num: {}, evader_num: {}".format(env.pursuer_num, env.evader_num))
	logger.info("pursuer_speed:{}, evader_speed:{}".format(env.pursuer_speed, env.evader_speed))
	logger.info('search_radius:{}'.format(env.search_radius))
	logger.info('evader_sec_radius:{}'.format(env.evader_sec_radius))
	# logger.info('pursuer_policy:{}'.format(env.pursuer_policy))
	logger.info('evader_policy:{}'.format(env.evader_policy))
	logger.info('communicate_radius:{}'.format(env.comm_radius))
	logger.info('ACTION INFO')
	logger.info('action: angular velocity')
	logger.info('action_dim:{}'.format(env.action_dim))
	if env.action_type == 'continue':
		logger.info('Use Continuous Action Space')
		logger.info("Action_Bound: {:.2f}".format(env.action_space.high))
	else:
		logger.info('Use Discrete Action Space')
		logger.info("Action_Bound: {:.2f}".format(env.max_action))
		logger.info("Action_Num: {:.2f}".format(env.action_space.n))



