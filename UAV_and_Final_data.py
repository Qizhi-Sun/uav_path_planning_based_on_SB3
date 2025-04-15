
# 记录各领导者-终点的匹配对
match_pairs_WH = [[0, [0, 0, 0], [45, 45, 6]]]

# 终点的x y z
x_goal = match_pairs_WH[0][2][0]
y_goal = match_pairs_WH[0][2][1]
z_goal = match_pairs_WH[0][2][2]

# 领导者的初始状态 x y z vx vy vz distance_to_final_x ..._y ..._z flag]
# flag 代表障碍物的标志位，周围有障碍物则置1，没有就是0
uav_init_pos_WH = [0, 0, 0, 0, 0, 0, x_goal, y_goal, z_goal, 0]


