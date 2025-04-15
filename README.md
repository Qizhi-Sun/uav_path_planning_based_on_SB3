# 中南大学交通运输工程学院轨道交通信号与控制专业2021级毕设
## 基于SB3框架实现的无人机集群路径规划

### 实现功能
- 领导者无人机寻路避障
- 跟随者无人机跟随领导者并保持初始编队
- 提供简单的继续训练方法与预训练好的checkpoints
- 提供基于matplotlib实现的测试过程渲染
- 提供可拓展与魔改的gym风格训练环境

### 快速开始
- 预训练模型存储于models文件夹
- 在main函数中加载预训练模型继续训练，或重新开始
- 在test_block中快速对比评估模型性能或通过render渲染测试画面

**执行训练**
```
python main.py
```
**执行测试**
````
python test_block_followers.py
````

### 参考项目

本项目的训练环境（包括那些奇怪的，点开就卡死的数据组织形式）来自项目：

<https://github.com/Lyytimmy/UAVGym.git>

### 后续工作
- 借助pygame实现更高性能的渲染
- 解决DDPG算法不收敛的问题


### 最后的最后

为什么要把这个很屎很垃圾的项目扔上来呢；  
一是证明我的毕设论文没有造假，经得起复现验证，毕设虽然水但还是尽我所能完成了一些工作的；  
二是希望能帮助到一些像我一样刚刚入门DRL的小白，学会怎么改环境，怎么用SB3快速出效果，但还请各位不要直接拿去当你的毕设（真的会有人偷垃圾吗），要恪守学术道德！；  
三是闲得无聊。  
顺带一提，要是不想用框架，作为造轮子享受者可以看看我同一个项目的另一个版本，有完整的网络搭建，梯度计算，参数更新过程，但是没收敛，嘻嘻。