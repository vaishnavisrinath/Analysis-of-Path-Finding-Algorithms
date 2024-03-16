import heapq
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dijkstra's Algorithm
def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances

# A* Search
def astar(graph, start, goal, num_nodes_per_side):
    open_list = [(0, start)]
    came_from = {}
    g_score = {node: float('infinity') for node in graph}
    g_score[start] = 0
    f_score = {node: float('infinity') for node in graph}
    f_score[start] = heuristic(start, goal, num_nodes_per_side)

    while open_list:
        current_node = min(open_list, key=lambda x: f_score[x[1]])
        open_list.remove(current_node)

        if current_node[1] == goal:
            return reconstruct_path(came_from, current_node[1])

        for neighbor, weight in graph[current_node[1]].items():
            tentative_g_score = g_score[current_node[1]] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node[1]
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, num_nodes_per_side)
                if neighbor not in [x[1] for x in open_list]:
                    open_list.append((f_score[neighbor], neighbor))

    return None

# Breadth-First Search
def bfs(graph, start, goal):
  queue = [(start, [start])]
  while queue:
    (node, path) = queue.pop(0)
    for next_node in graph[node]:
      if next_node == path[-1]:  # Check if next_node is the previous node
        continue
      new_path = list(path)
      new_path.append(next_node)
      queue.append((next_node, new_path))
      if next_node == goal:
        return new_path
  return None


# Depth-First Search
def dfs(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (node, path) = stack.pop()
        for next_node in graph[node]:
            if next_node not in path:
                new_path = path + [next_node]
                stack.append((next_node, new_path))
                if next_node == goal:
                    return new_path
    return None

# Helper function for A* heuristic (Manhattan distance for simplicity)
def heuristic(node, goal, num_nodes_per_side):
    x1, y1 = divmod(node, num_nodes_per_side)
    x2, y2 = divmod(goal, num_nodes_per_side)
    return abs(x1 - x2) + abs(y1 - y2)

# Helper function to reconstruct path for A*
def reconstruct_path(came_from, current_node):
    path = []
    while current_node in came_from:
        path.insert(0, current_node)
        current_node = came_from[current_node]
    return path

# Generate a star graph with specified number of nodes
def generate_star_graph(num_nodes):
   graph = {i: {} for i in range(num_nodes)}

   # Create directed edges from peripheral nodes to the central node
   for i in range(1, num_nodes):
       graph[i][0] = random.randint(1, 10)  # Edge going towards the central node

   return graph


# Measure the execution time of each algorithm for different input sizes
input_sizes =  [20000,50000,80000]
dijkstra_times = []
astar_times = []
bfs_times = []
dfs_times = []

for size in input_sizes:
    star_graph = generate_star_graph(size)

    start_time = time.time()
    dijkstra(star_graph, 0)
    dijkstra_time = time.time() - start_time
    dijkstra_times.append(dijkstra_time)

    start_time = time.time()
    astar(star_graph, 0, size - 1, int(size ** 0.5))
    astar_time = time.time() - start_time
    astar_times.append(astar_time)

    start_time = time.time()
    bfs(star_graph, 0, size - 1)
    bfs_time = time.time() - start_time
    bfs_times.append(bfs_time)

    start_time = time.time()
    dfs(star_graph, 0, size - 1)
    dfs_time = time.time() - start_time
    dfs_times.append(dfs_time)

for i in range(len(input_sizes)):
    print(f"Input Size: {input_sizes[i]}")
    print(f"Dijkstra's Algorithm: {dijkstra_times[i]} seconds")
    print(f"A* Search: {astar_times[i]} seconds")
    print(f"Breadth-First Search: {bfs_times[i]} seconds")
    print(f"Depth-First Search: {dfs_times[i]} seconds")
    print()


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(input_sizes, dijkstra_times, label='Dijkstra')
plt.plot(input_sizes, astar_times, label='A*')
plt.plot(input_sizes, bfs_times, label='BFS')
plt.plot(input_sizes, dfs_times, label='DFS')
plt.xlabel('Input Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison of Graph Search Algorithms (Star Graph)')
plt.legend()
plt.grid(True)
plt.show()
