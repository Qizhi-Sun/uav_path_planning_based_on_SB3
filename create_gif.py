import imageio
import os
import re

'''
用来把fig文件夹中的图像转化为gif,方便展示，哭了，我怎么这么贴心
'''
def create_gif(image_folder, output_gif, fps=10):
    """
    将文件夹中的图像文件转换为 GIF 动画

    :param image_folder: 包含图像文件的文件夹路径
    :param output_gif: 输出的 GIF 文件路径
    :param fps: 每秒帧数（帧率）
    """
    # 获取文件夹中所有图像文件并排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # 读取图像文件并添加到帧列表中
    frames = []
    for image in images:
        image_path = os.path.join(image_folder, image)
        frames.append(imageio.imread(image_path))

    # 将帧列表保存为 GIF 文件
    imageio.mimsave(output_gif, frames, fps=fps)


# 使用示例
image_folder = r'E:\RL\stable-baselin3\fig'  # 替换为你的图像文件夹路径
output_gif = 'output1.gif'  # 替换为你想要的输出 GIF 文件名
create_gif(image_folder, output_gif, fps=10)
