"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

# trajectory_generator.py
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, avg_speed=2.0):
        self.avg_speed = avg_speed

    def generate(self, path):
        """
        生成平滑轨迹
        :param path: N x 3 numpy array, 离散路径点
        :return: time_points, x_traj, y_traj, z_traj
        """
        path = np.array(path)
        x = path[:, 0]
        y = path[:, 1]
        z = path[:, 2]

        # 计算相邻点之间的距离
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
        # 累积距离得到总距离
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        # 根据平均速度计算时间戳
        times = cum_dist / self.avg_speed
        total_time = times[-1]

        # 创建三次样条插值函数
        cs_x = CubicSpline(times, x)
        cs_y = CubicSpline(times, y)
        cs_z = CubicSpline(times, z)

        # 生成密集的轨迹点用于绘图和输出
        t_dense = np.linspace(0, total_time, num=500)
        x_traj = cs_x(t_dense)
        y_traj = cs_y(t_dense)
        z_traj = cs_z(t_dense)

        return t_dense, x_traj, y_traj, z_traj, times, path

    def plot_trajectory(self, t_dense, x_traj, y_traj, z_traj, orig_times, orig_path):
        """
        绘制 x, y, z 关于时间的三个子图
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Plot X
        ax1.plot(t_dense, x_traj, label='Trajectory X')
        ax1.scatter(orig_times, orig_path[:, 0], c='r', marker='o', label='Path Points')
        ax1.set_ylabel('X (m)')
        ax1.set_title('Trajectory vs Time')
        ax1.legend()
        ax1.grid(True)

        # Plot Y
        ax2.plot(t_dense, y_traj, label='Trajectory Y', color='orange')
        ax2.scatter(orig_times, orig_path[:, 1], c='r', marker='o')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True)

        # Plot Z
        ax3.plot(t_dense, z_traj, label='Trajectory Z', color='green')
        ax3.scatter(orig_times, orig_path[:, 2], c='r', marker='o')
        ax3.set_ylabel('Z (m)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)

        # plt.show()