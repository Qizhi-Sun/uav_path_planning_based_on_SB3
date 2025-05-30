import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from UAV_Env_follower import *
from UAV_Env import UAVEnv

'''
用来测试领导者与跟随者1的情况，方便调试
'''
# 创建环境名
Map_name = 'Map1'
# 初始化MAP模块
MAP = SetConfig(Map_name)
uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
env_1 = UAVEnv(uav_num, map_w, map_h, map_z, Init_state, buildings)
train_env_1 = make_vec_env(lambda: env_1, n_envs=1)
model = SAC.load('E:\RL\stable-baselin3\models\save_3d_obstacle_22.zip', env=train_env_1)

# 初始化Env模块
env = UAVEnv_F(uav_num, map_w, map_h, map_z, Init_state, buildings, model)
train_env = make_vec_env(lambda: env, n_envs=1)

# 加载预训练模型
model_pre_trained = SAC.load('E:\RL\stable-baselin3\models\save_3d_follower_(0,0)_12_states.zip', env=train_env)
# 创建一个新的模型
# model_origin = SAC("MlpPolicy", train_env, verbose=0)

# state = train_env.reset()
# for i in range(100):
#     action = model.predict(state)[0]
#     next_state, r, done, truncated, info = env.step(action)
#     print(state, action, r)
#     state = next_state
#
#     if done:
#         break
render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool, match_pairs)
for i in range(1):
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model_pre_trained.predict(state)
        next_state, reward, done, truncated, _ = env.step(action)
        env_t = env.timestamp()
        env.recorder(env_t)
        render.render3D()
        plt.savefig(fr'E:\RL\stable-baselin3\fig\frame_{env_t}.png')
        plt.pause(0.01)
        state = next_state