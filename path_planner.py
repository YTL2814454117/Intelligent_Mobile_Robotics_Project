"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
import heapq
import math


class AStarPlanner:
    def __init__(self, env, resolution=1.0, heuristic_weight=2.0):
        self.env = env
        self.resolution = resolution
        self.w = heuristic_weight

        self.motions = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:
                        continue
                    cost = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                    self.motions.append(((x, y, z), cost))

    def plan(self, start, goal):
        print(f"[Planner] Start: {start}, Goal: {goal}")

        if self.env.is_collide(start):
            print("[Planner] Error: Start point is inside an obstacle!")
            return np.array([start])
        if self.env.is_collide(goal):
            print("[Planner] Error: Goal point is inside an obstacle!")
            return np.array([start])

        # 确保使用 tuple 以便哈希
        start_node = tuple(np.round(start, 2))
        goal_node = tuple(np.round(goal, 2))

        open_list = []
        heapq.heappush(open_list, (0, 0, start_node))

        came_from = {}
        g_score = {start_node: 0}

        iter_count = 0
        max_iter = 30000

        print("[Planner] Search started...")

        while open_list:
            iter_count += 1
            if iter_count > max_iter:
                print("[Planner] Timeout! Max iterations reached.")
                break

            if iter_count % 5000 == 0:
                print(f"[Planner] Iteration {iter_count}...")

            current_f, current_h, current = heapq.heappop(open_list)

            dist_to_goal = self.dist(current, goal_node)
            if dist_to_goal < self.resolution:
                print(f"[Planner] Goal reached in {iter_count} iterations!")
                # 防止自己指向自己的死循环
                if current != goal_node:
                    came_from[goal_node] = current
                return self.reconstruct_path(came_from, start_node, goal_node)

            for (dx, dy, dz), move_cost in self.motions:
                neighbor = (
                    round(current[0] + dx * self.resolution, 2),
                    round(current[1] + dy * self.resolution, 2),
                    round(current[2] + dz * self.resolution, 2)
                )

                if self.env.is_outside(neighbor):
                    continue

                new_g = g_score[current] + move_cost * self.resolution

                if neighbor not in g_score or new_g < g_score[neighbor]:
                    if self.env.is_collide(neighbor):
                        continue

                    g_score[neighbor] = new_g
                    h = self.dist(neighbor, goal_node)
                    f = new_g + self.w * h
                    heapq.heappush(open_list, (f, h, neighbor))
                    came_from[neighbor] = current

        print("[Planner] Failed to find a path.")
        return np.array([start])

    def dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def reconstruct_path(self, came_from, start, goal):
        path = []
        current = goal
        # 最大步数保护，防止万一出现的死循环卡死电脑
        safety_count = 0
        while current != start:
            path.append(current)
            current = came_from.get(current)

            if current is None:
                path.append(start)
                break

            safety_count += 1
            if safety_count > 10000:
                print("[Planner] Error: Infinite loop detected in path reconstruction!")
                break

        path.append(start)
        path.reverse()
        return np.array(path)



