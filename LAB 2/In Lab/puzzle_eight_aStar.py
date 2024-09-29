import heapq
import random
import time
import memory_profiler


# Author : Pratik Shah
# Date : Sept 4, 2024
# Place : IIIT Vadodara

# Course : CS307 Artificial Intelligence
# Exercise : Puzzle Eight Solver Using A Star


class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


def heuristic(state, goal_state):
    h = 0
    for i in range(1, 9):
        current_pos = state.index(i)
        goal_pos = goal_state.index(i)
        h += abs(current_pos // 3 - goal_pos // 3) + abs(current_pos % 3 - goal_pos % 3)
    return h


def get_successors(node):
    successors = []
    index = node.state.index(0)
    row, col = index // 3, index % 3

    moves = []
    if row > 0: moves.append(-3)  # Move up
    if row < 2: moves.append(3)  # Move down
    if col > 0: moves.append(-1)  # Move left
    if col < 2: moves.append(1)  # Move right

    # Generate new states based on valid moves
    for move in moves:
        new_index = index + move
        new_state = list(node.state)
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index]  # Swap blank with adjacent tile
        successor = Node(new_state, node, node.g + 1)  # Increase g (cost to reach this state)
        successors.append(successor)

    return successors


def search_agent(start_state, goal_state):
    """
    A* search algorithm that uses the heuristic and g (cost) to find the optimal path.
    """
    start_node = Node(start_state, g=0, h=heuristic(start_state, goal_state))  # Initialize start node
    frontier = []
    heapq.heappush(frontier, (start_node.f, start_node))  # Priority queue using f = g + h
    visited = set()  # Set to track visited states
    nodes_explored = 0

    while frontier:
        _, node = heapq.heappop(frontier)  # Get node with lowest f
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        nodes_explored += 1

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored:', nodes_explored)
            return path[::-1]

        for successor in get_successors(node):
            successor.h = heuristic(successor.state, goal_state)  # Update heuristic for successor
            successor.f = successor.g + successor.h  # Update f = g + h
            heapq.heappush(frontier, (successor.f, successor))  # Add successor to frontier

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

# Shuffle the puzzle
while d < D:
    successors = get_successors(s_node)
    s_node = random.choice(successors)  # Randomly pick a successor to shuffle the puzzle
    d += 1

goal_state = s_node.state

# Display start and goal states
print("Start State:")
print_puzzle(start_state)

print("Goal State:")
print_puzzle(goal_state)

# Measure time and memory usage
start_time = time.perf_counter()  # Use perf_counter for better accuracy

# Solve the puzzle using A* search
solution = search_agent(start_state, goal_state)

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
