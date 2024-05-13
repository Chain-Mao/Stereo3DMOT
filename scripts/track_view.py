# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from collections import defaultdict
# import numpy as np
# import matplotlib.lines as mlines
#
# def read_tracking_data(filename):
#     data = defaultdict(list)
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().split(' ')
#             frame, id, obj_type, _, _, _, _, _, _, _, _, _, _, x, y, z, _ = line
#             if obj_type == 'Car':  # 这里我们只关注车辆，如果你关注其他类型的对象，需要修改这里
#                 data[int(id)].append([float(x), float(z)])  # 注意这里我们取的是x和z，因为在KITTI坐标系中，x是向前，y是向左，z是向上，我们需要的鸟瞰图应当是x-z平面的视图
#     return data
#
# def draw_trajectory(data):
#     plt.figure()
#     lines = []
#     colors = {0: 'yellowgreen', 1: 'darkorange', 2: 'darkkhaki', 3: 'dodgerblue', 4: 'peru', 5: 'slategrey'}
#     for i, (id, trajectory) in enumerate(data.items()):
#         trajectory = np.array(trajectory)
#         color = colors.get(i, 'gray')
#         plt.plot(trajectory[:,0], trajectory[:,1], color=color, label=f'ID {id}', linewidth=3)
#         plt.scatter(trajectory[:,0], trajectory[:,1], color=color, s=30)
#         lines.append(mlines.Line2D([], [], color=colors[i], marker='o', markersize=5, label=f'ID {id}', linewidth=2))
#     plt.xlabel('X(m)')
#     plt.ylabel('Z(m)')
#     # 创建自定义图例
#     plt.legend(handles=lines)
#     plt.savefig('./groundtruth_trajectories', dpi=300)
#     plt.show()
#
#
# # 读取跟踪数据
# filename = '/home/simit/code/Stereo3DMOT/scripts/test.txt'  # 替换为你的实际文件名
# data = read_tracking_data(filename)
#
# # 绘制轨迹
# draw_trajectory(data)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import numpy as np
import matplotlib.lines as mlines

def read_tracking_data(filename):
    data = defaultdict(list)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            frame, id, obj_type, _, _, _, _, _, _, _, _, _, _, x, y, z, _, _ = line
            if obj_type == 'Car':  # 这里我们只关注车辆，如果你关注其他类型的对象，需要修改这里
                data[int(id)].append([float(x), float(z)])  # 注意这里我们取的是x和z，因为在KITTI坐标系中，x是向前，y是向左，z是向上，我们需要的鸟瞰图应当是x-z平面的视图
    return data

def draw_trajectory(data):
    plt.figure()
    lines = []
    colors = {0: 'darkkhaki', 1: 'darkorange', 2: 'yellowgreen', 3: 'dodgerblue', 4: 'peru', 5: 'slategrey'}
    for i, (id, trajectory) in enumerate(data.items()):
        trajectory = np.array(trajectory)
        color = colors.get(i, 'gray')
        plt.plot(trajectory[:,0], trajectory[:,1], color=color, label=f'ID {id}', linewidth=3)
        plt.scatter(trajectory[:,0], trajectory[:,1], color=color, s=30)
        lines.append(mlines.Line2D([], [], color=colors[i], marker='o', markersize=5, label=f'ID {id}', linewidth=2))
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    # 创建自定义图例
    plt.legend(handles=lines)
    plt.savefig('./stereo_trajectories', dpi=300)
    plt.show()


# 读取跟踪数据
filename = '/home/simit/code/Stereo3DMOT/scripts/test.txt'  # 替换为你的实际文件名
data = read_tracking_data(filename)

# 绘制轨迹
draw_trajectory(data)
