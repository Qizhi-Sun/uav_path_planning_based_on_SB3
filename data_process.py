import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # 使用 seaborn 的美化主题

leader_data_raw = np.loadtxt('episode_rewards.txt')
follower1_data_raw = np.loadtxt('episode_rewards_f1.txt')
follower2_data_raw = np.loadtxt("episode_rewards_f2.txt")

leader_data = leader_data_raw[:500]
follower1_data = follower1_data_raw[:500] + 5000
follower2_data = follower2_data_raw[:500] + 5000

x = range(len(leader_data))

plt.figure(figsize=(12, 6))
plt.plot(x, leader_data, label='Leader Reward', color='#FA7F6F', linestyle='-', linewidth=2.5)
plt.plot(x, follower1_data, label='Follower1 Reward', color='#FFBE7A', linestyle='-', linewidth=2.5)
plt.plot(x, follower2_data, label='Follower2 Reward', color='#8ECFC9', linestyle='-', linewidth=2.5)

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Episode Reward Curves', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('reward_curves.png', dpi=300, bbox_inches='tight')  # 可选：保存图片
plt.show()
