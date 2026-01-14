from flight_environment import FlightEnvironment
from path_planner import AStarPlanner
from trajectory_generator import TrajectoryGenerator
import numpy as np
import matplotlib.pyplot as plt
env = FlightEnvironment(50)
start = (1,2,0)
goal = (18,18,3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.
print("Planning Path...")
# 实例化 A* 规划器
planner = AStarPlanner(env, resolution=0.5, heuristic_weight=2.0)
path = planner.plan(start, goal)

print(f"Path Found! Nodes: {len(path)}")
# path = [[0,0,0],[1,1,1],[2,2,2],[3,3,3]]

# --------------------------------------------------------------------------------------------------- #


env.plot_cylinders(path)


# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

print("Generating Trajectory...")

# 实例化轨迹生成器，设置无人机平均速度
traj_gen = TrajectoryGenerator(avg_speed=3.0)

# 生成轨迹数据
t_dense, x_traj, y_traj, z_traj, orig_times, orig_path = traj_gen.generate(path)

# 2) 绘制轨迹曲线 (X-t, Y-t, Z-t)
traj_gen.plot_trajectory(t_dense, x_traj, y_traj, z_traj, orig_times, orig_path)
plt.show()

# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
