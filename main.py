import gymnasium as gym
from UAV_Env_follower import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from UAV_Env import  *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import PPO

# 创建环境名
Map_name = 'Map1'
# 初始化MAP模块
MAP = SetConfig(Map_name)
uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
# 初始化Env模块
env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state, buildings)
check_env(env, warn=True)
n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
env = Monitor(env)
train_env_1 = make_vec_env(lambda : env, n_envs=1)

'''
训练领导者
'''
# 新建一个SAC模型
# model = SAC(policy="MlpPolicy", env=train_env_1, verbose=1)
# model = SAC.load("E:\RL\stable-baselin3\models\save_3d_obstacle_21.zip", train_env_1, verbose=1)
# model.learn(total_timesteps=200000, progress_bar=True)
# model.save('E:\RL\stable-baselin3\models\save_3d_obstacle_22')


'''
训练跟随者1
'''
model_pre_trained_leader = SAC.load('E:\RL\stable-baselin3\models\save_3d_obstacle_22.zip', env=train_env_1)
# 初始化Env模块
env = UAVEnv_F(uav_num, map_w, map_h, map_z, Init_state, buildings, model_pre_trained_leader)
check_env(env, warn=True)
env = Monitor(env)
train_env = make_vec_env(lambda : env, n_envs=1)
# 新建一个SAC模型
# model = SAC(policy="MlpPolicy", env=train_env, verbose=1)
model = SAC.load("E:\RL\stable-baselin3\models\save_3d_follower_(4,0)_12_states.zip", env=train_env, verbose=1)
model.learn(total_timesteps=150000, progress_bar=True)
model.save('E:\RL\stable-baselin3\models\save_3d_follower_(4,0)1_12_states')

'''
训练跟随者2
'''
# model_pre_trained_leader = SAC.load('E:\RL\stable-baselin3\models\save_3d_obstacle_22.zip', env=train_env_1)
# # 初始化Env模块
# env = UAVEnv_F(uav_num, map_w, map_h, map_z, Init_state, buildings, model_pre_trained_leader)
# check_env(env, warn=True)
# train_env = make_vec_env(lambda : env, n_envs=1)
# model = SAC.load("E:\RL\stable-baselin3\models\save_3d_follower_(4,0)_12_states.zip", env=train_env, verbose=1)
# model.learn(total_timesteps=100000, progress_bar=True)
# model.save('E:\RL\stable-baselin3\models\save_3d_follower_(0,0)_12_states')

'''
记录存储reward
'''
# episode_rewards = train_env_1.envs[0].get_episode_rewards()
# np.savetxt('episode_rewards_L_PPO.txt', episode_rewards, fmt='%f', delimiter=',')
# plt.plot(episode_rewards)
# plt.title("Follower_2 Episode Rewards")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()