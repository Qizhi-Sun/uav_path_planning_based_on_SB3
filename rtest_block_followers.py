import math
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from UAV_Env_follower import UAVEnv_F
from UAV_Env import UAVEnv
from UAV_Env_follower_2 import *
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# 测试回合数
episode_num = 1
# 指标初始化
js = 0
jc = 0
j_mcr = 0
j_fkr = 0
# 高斯噪声幅度
scale_1 = 0.001
scale_2 = 0.002
scale_3 = 0.003
scale_4 = 0.01
'''
用来测试领导者与跟随者1，2的情况
'''
# 创建环境名
Map_name = 'Map1'
# 初始化MAP模块
MAP = SetConfig(Map_name)
uav_num, map_w, map_h, map_z, buildings_location, buildings, match_pairs, uav_r, Init_state = MAP.Setting()
env_1 = UAVEnv(uav_num, map_w, map_h, map_z, Init_state, buildings)
train_env_1 = make_vec_env(lambda: env_1, n_envs=1)
model = SAC.load('E:\RL\stable-baselin3\models\save_3d_obstacle_22.zip', env=train_env_1)
# model = SAC(policy="MlpPolicy", env=train_env_1, verbose=1)
# 初始化Env模块
env = UAVEnv_F(uav_num, map_w, map_h, map_z, Init_state, buildings, model)
# env = GoLeftEnv()
train_env = make_vec_env(lambda: env, n_envs=1)

# 加载预训练模型
model_2 = SAC.load('E:\RL\stable-baselin3\models\save_3d_follower_(4,0)1_12_states.zip', env=train_env)
# model_2 = SAC(policy="MlpPolicy", env=train_env, verbose=1)
model_3 = [model, model_2]

env_2 = UAVEnv_F2(uav_num, map_w, map_h, map_z, Init_state, buildings, model_3)
train_env_2 = make_vec_env(lambda: env_2, n_envs=1)
model_pre_trained = SAC.load('E:\RL\stable-baselin3\models\save_3d_follower_(0,0)_12_states.zip', env=train_env_2)
# model_pre_trained = SAC(policy="MlpPolicy", env=train_env_2, verbose=1)
# 创建一个新的模型
# model_origin = SAC("MlpPolicy", train_env, verbose=0)
render = Render(uav_num, buildings, map_w, map_h, map_z, uav_r, env_2.position_pool, match_pairs)

for i in range(episode_num):
    state, _ = env_2.reset()
    done = False
    truncated = False
    dis = 0
    c_sum = []
    mkr = 0
    while not (done or truncated):
        action, _ = model_pre_trained.predict(state)
        # action += np.random.normal(loc=0, scale=scale_4, size=len(action)) # 鲁棒性测试
        next_state, reward, done, truncated, info = env_2.step(action)
        env_t = env_2.timestamp()
        env_2.recorder(env_t)
        render.render3D()
        plt.savefig(fr'E:\RL\stable-baselin3\change_team\frame_{env_t}.png')
        plt.pause(0.01)
        x_diff = abs(next_state[0] - state[0])
        y_diff = abs(next_state[1] - state[1])
        z_diff = abs(next_state[2] - state[2])
        dis_diff = math.hypot(x_diff, y_diff, z_diff)
        dis += dis_diff * 10
        action_diff = math.hypot(action[0], action[1], action[2]-0.01)# 减去重力加速度
        c_sum.append(action_diff * 0.1 * 0.5) # 0.1为dt 0.5是因为这里加速度范围用了-30-30
        state = next_state
    if done and info:
        mkr = 1




    # leader
    xyz = env_2.position_pool[1]
    x = [(sublist[0]-45) * 10 for sublist in xyz]
    y = [(sublist[1]-45) * 10 for sublist in xyz]
    z = [(sublist[2]-6) * 10 for sublist in xyz]
    t = [sublist[3] * 0.2 for sublist in xyz] #这里为什么×0.2呢 因为recoder函数中是每两步记录一次渲染数据 dt=0.1

    # follower(0 0 2)
    xyz_f1 = env_2.position_pool[0]
    x_f1 = [sublist[0] * 10  for sublist in xyz_f1]
    x_f_1 = [(sublist[0] * 10 - 450)  for sublist in xyz_f1]
    x_f11 = [abs(x_f1[i] - val - 450 + 20) * 0.5 for i, val in enumerate(x)]
    y_f1 = [sublist[1] * 10 for sublist in xyz_f1]
    y_f_1 = [(sublist[1] * 10 - 450)  for sublist in xyz_f1]
    y_f11 = [abs(y_f1[i] - val - 450 + 20) * 0.5 for i, val in enumerate(y)]
    z_f1 = [sublist[2] * 10 for sublist in xyz_f1]
    z_f_1 = [(sublist[2] * 10 - 60)  for sublist in xyz_f1]
    z_f11 = [abs(z_f1[i] - val - 60) * 0.5 for i, val in enumerate(z)]
    t_f1 = [sublist[3] * 0.2 for sublist in xyz_f1]


    # follower(4 0 2)
    xyz_f2 = env_2.position_pool[2]
    x_f2 = [sublist[0] * 10 for sublist in xyz_f2]
    x_f_2 = [(sublist[0]-45) * 10 for sublist in xyz_f2]
    x_f22 = [abs(x_f2[i] - val -450 - 20) * 0.175 for i, val in enumerate(x)]
    y_f2 = [sublist[1] * 10 for sublist in xyz_f2]
    y_f_2 = [(sublist[1]-45) * 10 for sublist in xyz_f2]
    y_f22 = [abs(y_f2[i] - val -450 + 20) * 0.175 for i, val in enumerate(y)]
    z_f2 = [sublist[2] * 10 for sublist in xyz_f2]
    z_f_2 = [(sublist[2]-6) * 10 for sublist in xyz_f2]
    z_f22 = [abs(z_f2[i] - val - 60) * 0.175 for i, val in enumerate(z)]
    t_f2 = [sublist[3] * 0.2 for sublist in xyz_f1]

    plt.figure(100000000, figsize=(12,6))
    plt.plot(t, x , label='X坐标差', color='#4C72B0', lw = 2)  # Seaborn 调色板中的颜色
    plt.plot(t, y , label='Y坐标差', color='#DD8452', lw = 2)  # Seaborn 调色板中的颜色
    plt.plot(t, z , label='Z坐标差', color='#55A868', lw = 2)  # Seaborn 调色板中的颜色
    dis_error = [math.hypot(x[i], y[i], z[i]) for i in range(len(t))]
    plt.plot(t, dis_error, label="距离", color='#FFBE7A', lw = 2)
    plt.xlabel('飞行时间/s', fontsize=16)
    plt.ylabel('距终点位置差/m', fontsize=16)
    plt.title('领导者位置偏差随时间变化', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pos_error_leader.png', dpi=1200, bbox_inches='tight')  # 可选：保存图片
    plt.show()


    plt.figure(100000001, figsize=(6,4))
    plt.plot(t, x_f_1, label='X坐标差', color='#4C72B0', lw = 2)  # Seaborn 调色板中的颜色
    plt.plot(t, y_f_1, label='Y坐标差', color='#DD8452', lw = 2)  # Seaborn 调色板中的颜色
    plt.plot(t, z_f_1, label='Z坐标差', color='#55A868', lw = 2)  # Seaborn 调色板中的颜色
    dis_error = [math.hypot(x_f_1[i], y_f_1[i], z_f_1[i]) for i in range(len(t))]
    plt.plot(t, dis_error, label="距离", color='#FFBE7A', lw = 2)
    plt.xlabel('飞行时间/s',fontsize=16)
    plt.ylabel('距终点位置差/m',fontsize=16)
    plt.title('跟随者1位置偏差随时间变化',fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pos_error_f1.png', dpi=1200, bbox_inches='tight')  # 可选：保存图片
    plt.show()

    plt.figure(100000002, figsize=(6,4))
    plt.plot(t, x_f_2, label='X坐标差', color='#4C72B0', lw = 2)  # Seaborn 调色板中的颜色
    plt.plot(t, y_f_2, label='Y坐标差', color='#DD8452', lw = 2)  # Seaborn 调色板中的颜色
    plt.plot(t, z_f_2, label='Z坐标差', color='#55A868', lw = 2)  # Seaborn 调色板中的颜色
    dis_error = [math.hypot(x_f_2[i], y_f_2[i], z_f_2[i]) for i in range(len(t))]
    plt.plot(t, dis_error, label="距离", color='#FFBE7A', lw = 2)
    plt.xlabel('飞行时间/s', fontsize=16)
    plt.ylabel('距终点位置差/m', fontsize=16)
    plt.title('跟随者2位置偏差随时间变化', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pos_error_f2.png', dpi=1200, bbox_inches='tight')  # 可选：保存图片
    plt.show()

    plt.figure(100000003)
    plt.plot(t_f1, x_f11, label='X坐标差', color='#4C72B0', lw=2)  # Seaborn 调色板中的颜色
    plt.plot(t_f1, y_f11, label='Y坐标差', color='#DD8452', lw=2)  # Seaborn 调色板中的颜色
    plt.plot(t_f1, z_f11, label='Z坐标差', color='#55A868', lw=2)  # Seaborn 调色板中的颜色
    dis_error_f1 = [math.hypot(x_f11[i], y_f11[i], z_f11[i]) for i in range(len(t))]
    num = 0
    for i in range(len(t)):
        if dis_error_f1[i] <= 2:
            num += 1
    fkr = num / len(t)
    plt.plot(t, dis_error_f1, label="距离", color='#FFBE7A', lw = 2)
    plt.xlabel('飞行时间/s', fontsize=12)
    plt.ylabel('位置偏差/m', fontsize=12)
    plt.title('跟随者1位置偏差随时间变化',fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.ylim(0,3)
    plt.tight_layout()
    plt.savefig('f1.png', dpi=1200, bbox_inches='tight')  # 可选：保存图片
    plt.show()

    plt.figure(100000004)
    plt.plot(t_f2, x_f22, label='X坐标差', color='#4C72B0', lw=2)  # Seaborn 调色板中的颜色
    plt.plot(t_f2, y_f22, label='Y坐标差', color='#DD8452', lw=2)  # Seaborn 调色板中的颜色
    plt.plot(t_f2, z_f22, label='Z坐标差', color='#55A868', lw=2)  # Seaborn 调色板中的颜色
    dis_error_f2 = [math.hypot(x_f22[i], y_f22[i], z_f22[i]) for i in range(len(t))]
    plt.plot(t, dis_error_f2, label="距离", color='#FFBE7A', lw = 2)
    plt.xlabel('飞行时间/s', fontsize=12)
    plt.ylabel('位置偏差/m', fontsize=12)
    plt.title('跟随者2位置偏差随时间变化',fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.ylim(0,3)
    plt.tight_layout()
    plt.savefig('f2.png', dpi=1200, bbox_inches='tight')  # 可选：保存图片
    plt.show()

    # 指标js
    js += dis

    # 指标jc
    jc += sum(c_sum) * 1000 # 单位换算 1单位-》1000m/s**2；

    # 指标j_mcr
    j_mcr += mkr

    # 指标j_fkr
    j_fkr += fkr

print(f"js is {js / episode_num}")
print(f"jc is {jc / episode_num}")
print(f"j_mcr is {j_mcr / episode_num}")
print(f"j_fkr is {j_fkr / episode_num}")