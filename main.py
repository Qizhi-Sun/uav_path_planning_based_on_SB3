import gymnasium as gym
from num_env import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from UAV_Env import  *

# 创建环境名
Map_name = 'Map1'
# 初始化MAP模块
MAP = SetConfig(Map_name)
uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
# 初始化Env模块
env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state, buildings)
# env = GoLeftEnv()
check_env(env, warn=True)
train_env = make_vec_env(lambda : env, n_envs=1)
model = SAC("MlpPolicy", train_env, verbose=1)
# model = A2C.load('E:\RL\stable-baselin3\models\save_3d_3.zip', env=train_env)
model.learn(total_timesteps=600000, progress_bar=True)
model.save('E:\RL\stable-baselin3\models\save_3d_obstacle_12')
