import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

leader_data_raw = np.loadtxt('episode_rewards.txt')
follower1_data_raw = np.loadtxt('episode_rewards_f1.txt')
follower2_data_raw = np.loadtxt("episode_rewards_f2.txt")

leader_data = leader_data_raw[:500]
follower1_data = follower1_data_raw[:500] + 5000
follower2_data = follower2_data_raw[:500] + 5000

x = range(len(leader_data))

plt.figure(figsize=(12, 6))
plt.plot(x, leader_data, label='领导者回合奖励', color='#FA7F6F', linestyle='-', linewidth=2)
plt.plot(x, follower1_data, label='跟随者1 回合奖励', color='#FFBE7A', linestyle='-', linewidth=2)
plt.plot(x, follower2_data, label='跟随者2 回合奖励', color='#8ECFC9', linestyle='-', linewidth=2)

plt.xlabel('回合数', fontsize=16)
plt.ylabel('回合总奖励', fontsize=16)
plt.title('回合累积奖励曲线', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.8)

plt.tight_layout()
plt.savefig('reward_curves.png', dpi=1200, bbox_inches='tight')  # 可选：保存图片
plt.show()
