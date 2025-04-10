import gymnasium as gym
from UAV_Env_follower import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from UAV_Env import  *

'''
训练领导者
'''
# 创建环境名
Map_name = 'Map1'
# 初始化MAP模块
MAP = SetConfig(Map_name)
uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
# 初始化Env模块
env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state, buildings)
check_env(env, warn=True)
train_env_1 = make_vec_env(lambda : env, n_envs=1)
# model = SAC.load("E:\RL\stable-baselin3\models\save_3d_obstacle_13.zip", train_env_1, verbose=1)
# model.learn(total_timesteps=800000, progress_bar=True)
# model.save('E:\RL\stable-baselin3\models\save_3d_obstacle_14')

'''
训练跟随者
'''
model_pre_trained_leader = SAC.load('E:\RL\stable-baselin3\models\save_3d_obstacle_14.zip', env=train_env_1)
# 初始化Env模块
env = UAVEnv_F(uav_num, map_w, map_h, map_z, Init_state, buildings, model_pre_trained_leader)
check_env(env, warn=True)
train_env = make_vec_env(lambda : env, n_envs=1)
model = SAC.load("E:\RL\stable-baselin3\models\save_3d_follower_3.zip", env=train_env, verbose=1)
model.learn(total_timesteps=80000, progress_bar=True)
model.save('E:\RL\stable-baselin3\models\save_3d_follower_4')