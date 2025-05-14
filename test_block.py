import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from UAV_Env import *

'''
用来单独测试领导者的避障与寻路，方便调试
'''

# 创建环境名
Map_name = 'Map1'
# 初始化MAP模块
MAP = SetConfig(Map_name)
uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
# 初始化Env模块
env = UAVEnv(uav_num, map_w, map_h, map_z, Init_state, buildings)
# env = GoLeftEnv()
train_env = make_vec_env(lambda: env, n_envs=1)

# 加载预训练模型
model_pre_trained = SAC.load('E:\RL\stable-baselin3\models\save_3d_obstacle_21.zip', env=train_env)


render = Render(uav_num, env.state, buildings, map_w, map_h, map_z, uav_r, env.position_pool, match_pairs)
num_success = 0
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
        # plt.savefig(fr'E:\RL\stable-baselin3\fig_w2\frame_{env_t}.png')
        plt.pause(0.01)
        state = next_state
    if done:
        num_success += 1

print(f"j_mcr is {num_success}")