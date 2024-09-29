from collections import deque
import random
import time
import memory_profiler


# Author : Pratik Shah
# Date : August 20, 2024
# Place : IIIT Vadodara

# Course : CS307 Artificial Intelligence
# Exercise : Puzzle Eight Solver Using BFS


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


def get_successors(node):
    successors = []
    index = node.state.index(0)  # Find the blank (0) position
    row, col = index // 3, index % 3  # Get the row and column of the blank space

    # Possible moves (up, down, left, right) constrained by the position of the blank
    moves = []
    if row > 0:
        moves.append(-3)  # Move up
    if row < 2:
        moves.append(3)  # Move down
    if col > 0:
        moves.append(-1)  # Move left
    if col < 2:
        moves.append(1)  # Move right

    # Generate new states for each valid move
    for move in moves:
        new_index = index + move
        new_state = list(node.state)
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index]  # Swap blank with adjacent tile
        successors.append(Node(new_state, node))

    return successors


def bfs(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0

    while queue:
        node = queue.popleft()

        if tuple(node.state) in visited:
            continue

        visited.add(tuple(node.state))
        nodes_explored += 1

        if node.state == goal_node.state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored:', nodes_explored)
            return path[::-1]  # Return the path from start to goal

        for successor in get_successors(node):
            queue.append(successor)

    return None  # No solution found


def print_puzzle(state):
    for i in range(0, 9, 3):
        print(state[i:i + 3])
    print()


# Initialize start state and shuffle to generate goal state
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
s_node = Node(start_state)
D = 20  # Number of random moves to shuffle the puzzle
d = 0

while d < D:
    successors = get_successors(s_node)
    s_node = random.choice(successors)  # Choose a random valid successor state
    d += 1

goal_state = s_node.state

# Display start and goal states
print("Start State:")
print_puzzle(start_state)

print("Goal State:")
print_puzzle(goal_state)

# Measure time and memory usage
start_time = time.perf_counter()  # Use perf_counter for better accuracy

# Solve the puzzle using BFS
solution = bfs(start_state, goal_state)

end_time = time.perf_counter()

# Output the solution path
if solution:
    print("Solution found:")
    for step in solution:
        print_puzzle(step)
else:
    print("No solution found.")

# Print time and memory usage
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.6f} seconds")

# Use memory profiler to estimate memory usage
mem_usage = memory_profiler.memory_usage(-1, interval=0.1, timeout=1)
print(f"Memory usage (MiB): {max(mem_usage) - min(mem_usage):.6f}")
