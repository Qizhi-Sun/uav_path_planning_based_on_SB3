import random
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from building_data import *
from UAV_and_Final_data import *
import matplotlib.style as mplstyle

'''
领导者的训练环境
'''

mplstyle.use('fast')
x_goal = match_pairs_WH[0][2][0]
y_goal = match_pairs_WH[0][2][1]
z_goal = match_pairs_WH[0][2][2]

# 初始化无人机环境
class UAVEnv(gym.Env):
    def __init__(self, uav_num, map_w, map_h, map_z, Init_state, buildings):
        super().__init__()
        self.uav_num = uav_num
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.position_pool = [[] for _ in range(self.uav_num)]
        self.state = Init_state
        self.buildings = buildings
        self.info = {}
        self.r = [0 for _ in range(self.uav_num)]
        self.done = False
        self.truncated = False
        self.env_t = 0
        # 定义无人机的动作空间和观测空间
        self.action_space = spaces.Box(low=np.array([-0.03, -0.03, -0.03] * self.uav_num),
                                       high=np.array([0.03, 0.03, 0.03] * self.uav_num), dtype=np.float32)
        # 状态包括x y z vx vy vz xg yg zg o 这里速度限制是0.2而非0.4,在step函数里进行了clip限制,这里写0.4单纯是方便调整测试.
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -0.4, -0.4, -0.4, -self.map_w, -self.map_h, -self.map_z, 0] * self.uav_num),
                                            high=np.array([self.map_w, self.map_h, self.map_z, 0.4, 0.4, 0.4, self.map_w, self.map_h, self.map_z, 1] *
                                                          self.uav_num), dtype=np.float32)

    # 记录无人机的飞行轨迹函数
    def recorder(self, env_t):
        if env_t % 2 == 1:
            for i in range(self.uav_num):
                x, y, z = self.state[:3]
                position = [x, y, z, env_t]
                self.position_pool[i].append(position)

    # 无人机的动作更新函数
    def step(self, actions):
        self.env_t += 1
        last_speed = self.state[3:6]
        self.state[3:6] +=actions[:3]
        self.state[3:6] = np.clip(self.state[3:6], -0.2, 0.2)
        # update x, y, z
        for i in range(3):
            dis = (last_speed[i] + self.state[i+3]) / 2
            self.state[i] += dis


        '''
        r_edge
        '''
        if self.state[0]<1 or self.state[0]>49 or self.state[1]<1 or self.state[1]>49 or self.state[2]<1 or self.state[2]>9:
            r_edge = -5
        elif self.state[0]<=0 or self.state[0]>=50 or self.state[1]<=0 or self.state[1]>=50 or self.state[2]<=0 or self.state[2]>=10:
            r_edge = -10
        else:
            r_edge = 0
        self.state[:2] = np.clip(self.state[:2], 0, 49.9)
        self.state[2] = np.clip(self.state[2], 0, 10)

        '''
        r_obstacle
        '''
        pos_x_diff_1 = self.state[0] - 15 # uav[x] - obstacle_x
        pos_y_diff_1 = self.state[1] - 15 # uav[y] - obstacle_y
        pos_x_diff_2 = self.state[0] - 25 # uav[x] - obstacle_x
        pos_y_diff_2 = self.state[1] - 25 # uav[y] - obstacle_y
        pos_x_diff_3 = self.state[0] - 35 # uav[x] - obstacle_x
        pos_y_diff_3 = self.state[1] - 35 # uav[y] - obstacle_y
        distance_to_obstacle1 = math.hypot(pos_x_diff_1, pos_y_diff_1) # 距离障碍物中心的距离1
        distance_to_obstacle2 = math.hypot(pos_x_diff_2, pos_y_diff_2) # 距离障碍物中心的距离2
        distance_to_obstacle3 = math.hypot(pos_x_diff_3, pos_y_diff_3) # 距离障碍物中心的距离3
        min_distance_to_obs = min(distance_to_obstacle3, distance_to_obstacle2, distance_to_obstacle1)
        if min_distance_to_obs <= 8:
            r_obstacle_1 = -(10 - min_distance_to_obs) * 5
            self.state[9] = 1   # 更新障碍物标志位 obstacle_flag = 1 代表无人机附近有障碍物
        else:
            r_obstacle_1 = 0
        grid_x = int(self.state[0])
        grid_y = int(self.state[1])
        if self.buildings[grid_x * 50 + grid_y][4] == 2:
            height = 10
        elif self.buildings[grid_x * 50 + grid_y][4] == 3:
            height = 10
        else:
            height = 0
        if self.state[2] <= height:
            r_obstacle = -500
            self.done = True
        else:
            r_obstacle = r_obstacle_1


        '''
        r_goal
        '''
        x_diff = self.state[0]-x_goal
        y_diff = self.state[1]-y_goal
        z_diff = self.state[2]-z_goal
        last_distance = math.hypot(self.state[6], self.state[7], self.state[8])
        self.state[6] = x_diff
        self.state[7] = y_diff
        self.state[8] = z_diff
        distance_to_goal = math.hypot(x_diff, y_diff, z_diff)

        if distance_to_goal <= 2:
            self.done = True
            r_goal = 500
            print("!!!!!!!!!!!!!!! UAV HAVE BEEN FINAL !!!!!!!!!!!!!!!")
        else:
            # r_goal = -0.05 * distance_to_goal
            r_goal = 100 * (last_distance - distance_to_goal)

        if self.env_t >= 500:
            self.truncated = True


        self.r = r_goal + r_edge + r_obstacle
        # self.r = r_goal + r_edge
        return np.array(self.state, dtype=np.float32), float(self.r), self.done, self.truncated, self.info

    def reset(self, seed = None):
        self.state =[2, 2, 2, 0, 0, 0, 2-x_goal, 2-y_goal, 2-z_goal, 0]
        self.r = 0
        self.done = False
        self.truncated = False
        self.env_t = 0
        return np.array(self.state, dtype=np.float32), self.info

    def timestamp(self):
        return self.env_t

# 画面渲染函数，使用matplotlib库绘制地图、障碍物、无人机;这效率确实有点低,改用pygame好一点.
class Render:
    def __init__(self, uav_num, state, buildings, map_w, map_h, map_z, uav_r, position_pool, match_pairs):
        self.uav_num = uav_num
        self.state = state
        self.buildings = buildings
        self.map_w = map_w
        self.map_h = map_h
        self.map_z = map_z
        self.uav_r = uav_r
        self.position_pool = position_pool
        self.line = []
        self.match_pairs = match_pairs
        self.AimsPoint = [[] for _ in range(self.uav_num)]
        self.Head = []

        # 创建画布
        self.fig = plt.figure(figsize=(self.map_w, self.map_h))  # 设置画布大小
        self.ax = self.fig.add_subplot(111, projection='3d')  # 创建三维坐标系
        self.ax.view_init(elev=90, azim=0)
        # 绘制目标点
        for index, pair in enumerate(match_pairs):
            aim = pair[2]
            Point = self.ax.scatter(aim[0], aim[1], aim[2], color='deepskyblue', s=20)
            self.AimsPoint[index].append(Point)
        # 绘制建筑
        for building in self.buildings:
            x = [building[0][0], building[1][0], building[3][0], building[2][0]]
            y = [building[0][1], building[1][1], building[3][1], building[2][1]]
            z = [building[0][2], building[1][2], building[3][2], building[2][2]]
            building_type = building[4]

            if building_type == 0:
                continue

            if building_type == 1:
                height = 4
                color = 'lightgreen'
            elif building_type == 2:
                height = 10
                color = 'lightblue'
            elif building_type == 3:
                height = 10
                color = 'lightblue'

            vertices = [
                [x[0], y[0], z[0]],
                [x[1], y[1], z[1]],
                [x[2], y[2], z[2]],
                [x[3], y[3], z[3]],
                [x[0], y[0], z[0] + height],
                [x[1], y[1], z[1] + height],
                [x[2], y[2], z[2] + height],
                [x[3], y[3], z[3] + height]
            ]
            faces = [
                # [0, 1, 2, 3],  # bottom face 不打印底面减小性能开销
                [4, 5, 6, 7],  # top face
                [0, 1, 5, 4],  # front face
                [1, 2, 6, 5],  # right face
                [2, 3, 7, 6],  # back face
                [3, 0, 4, 7]  # left face
            ]

            cuboid = Poly3DCollection([[vertices[point] for point in face] for face in faces], facecolors=color,
                                      linewidths=0.5, edgecolors='gray', alpha=1)
            self.ax.add_collection3d(cuboid)

        self.ax.set_xlim(0, map_w + 1)
        self.ax.set_ylim(0, map_h + 1)
        self.ax.set_zlim(0, map_z + 1)

    # 绘制无人机
    def render3D(self):
        plt.ion()
        for i in range(self.uav_num):
            x_traj, y_traj, z_traj, _ = zip(*self.position_pool[i])
            if i == 0:
                l = self.ax.plot(x_traj[-10:], y_traj[-10:], z_traj[-10:], color='gray', alpha=0.7, linewidth=2.0)
                self.line.append(l)
                head = self.ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='darkorange', s=30)
                self.Head.append(head)
            if i == 1:
                l = self.ax.plot(x_traj[-10:], y_traj[-10:], z_traj[-10:], color='gray', alpha=0.7, linewidth=2.0)
                self.line.append(l)
                head = self.ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='deepskyblue', s=30)
                self.Head.append(head)
            if i == 2:
                l = self.ax.plot(x_traj[-10:], y_traj[-10:], z_traj[-10:], color='gray', alpha=0.7, linewidth=2.0)
                self.line.append(l)
                head = self.ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='gray', s=30)
                self.Head.append(head)
        # 更新轨迹和无人机本体位置
        while len(self.line) > self.uav_num:
            old_line = self.line.pop(0)
            old_line[0].remove()
        while len(self.Head) > self.uav_num:
            old_head = self.Head.pop(0)
            old_head.remove()


# 参数配置，目前可供选择的演示地图有Map1、Map2
class SetConfig:
    def __init__(self, name):
        self.name = name
        self.uav_num = 0
        self.uav_r = 0.3
        self.map_w, self.map_h, self.map_z = 0, 0, 0
        self.buildings_location = []
        self.buildings = []
        self.match_pairs = []
        self.Init_state = []

    def Setting(self):
        if self.name == 'Map1':
            self.uav_num = 1
            self.map_w, self.map_h, self.map_z = 50, 50, 10
            self.buildings_location = buildings_location_WH
            self.buildings = buildings_WH
            self.match_pairs = match_pairs_WH
            self.Init_state = uav_init_pos_WH
            for i in range(2500):
                self.buildings[i][4] = 0
            # 地图2 W2型地图
            # for i in range(13,17):
            #     for j in range(13,17):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(23,27):
            #     for j in range(3,7):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(3,7):
            #     for j in range(23,27):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(33,37):
            #     for j in range(33,37):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(43,47):
            #     for j in range(23,27):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(23,27):
            #     for j in range(43,47):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(23,27):
            #     for j in range(13,17):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(13,17):
            #     for j in range(23,27):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(33,37):
            #     for j in range(23,27):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(23,27):
            #     for j in range(33,37):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2

            # 地图1 九宫格型地图
            for i in range(13,17):
                for j in range(13,17):
                    idx = j * 50
                    self.buildings[i+idx][4] = 2
            for i in range(23,27):
                for j in range(23,27):
                    idx = j * 50
                    self.buildings[i+idx][4] = 2
            for i in range(33,37):
                for j in range(33,37):
                    idx = j * 50
                    self.buildings[i+idx][4] = 2

            for i in range(13,17):
                for j in range(23,27):
                    idx = j * 50
                    self.buildings[i+idx][4] = 1
            for i in range(13,17):
                for j in range(33,37):
                    idx = j * 50
                    self.buildings[i+idx][4] = 2
            for i in range(23,27):
                for j in range(33,37):
                    idx = j * 50
                    self.buildings[i+idx][4] = 1

            for i in range(23,27):
                for j in range(13,17):
                    idx = j * 50
                    self.buildings[i+idx][4] = 1
            for i in range(33,37):
                for j in range(13,17):
                    idx = j * 50
                    self.buildings[i+idx][4] = 2
            for i in range(33,37):
                for j in range(23,27):
                    idx = j * 50
                    self.buildings[i+idx][4] = 1

            # 地图三 三柱地图
            # for i in range(10,20):
            #     for j in range(10,20):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(20,30):
            #     for j in range(20,30):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
            # for i in range(30,40):
            #     for j in range(30,40):
            #         idx = j * 50
            #         self.buildings[i+idx][4] = 2
        else:
            print("参数错误")
            sys.exit()

        return self.uav_num, self.map_w, self.map_h, self.map_z, self.buildings_location, self.buildings, self.match_pairs, self.uav_r, self.Init_state



