from collections import deque
import time


# Check if the current state is the goal state
def is_goal(state):
    return state == [1, 1, 1, 0, -1, -1, -1]


# Generate all possible successor states
def get_successors(state):
    successors = []
    empty_index = state.index(0)  # 0 represents the empty space
    moves = [-1, -2, 1, 2]  # Possible moves: swap with adjacent or 2 spaces away

    for move in moves:
        new_index = empty_index + move
        if 0 <= new_index < len(state):  # Ensure new index is within bounds
            new_state = state[:]
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            successors.append(new_state)

    return successors


# Breadth-First Search (BFS) to find the solution
def bfs(start_state):
    queue = deque([(start_state, [])])
    visited = set()
    nodes_visited = 0
    max_memory = 0
    start_time = time.perf_counter()  # Start timing for BFS
    while queue:
        current_state, path = queue.popleft()
        nodes_visited += 1
        if tuple(current_state) in visited:
            continue
        visited.add(tuple(current_state))
        path = path + [current_state]

        # Update maximum memory usage
        max_memory = max(max_memory, len(queue) + len(visited))

        if is_goal(current_state):
            elapsed_time = time.perf_counter() - start_time
            return path, nodes_visited, max_memory, elapsed_time

        for successor in get_successors(current_state):
            queue.append((successor, path))

    elapsed_time = time.perf_counter() - start_time
    return None, nodes_visited, max_memory, elapsed_time


# Depth-First Search (DFS) to find the solution
def dfs(start_state):
    stack = [(start_state, [])]  # Use a stack for DFS
    visited = set()
    nodes_visited = 0
    max_memory = 0
    start_time = time.perf_counter()  # Start timing for DFS
    while stack:
        current_state, path = stack.pop()  # Pop from the end of the list (LIFO)
        nodes_visited += 1
        if tuple(current_state) in visited:
            continue
        visited.add(tuple(current_state))
        path = path + [current_state]

        # Update maximum memory usage
        max_memory = max(max_memory, len(stack) + len(visited))

        if is_goal(current_state):
            elapsed_time = time.perf_counter() - start_time
            return path, nodes_visited, max_memory, elapsed_time

        for successor in get_successors(current_state):
            stack.append((successor, path))

    elapsed_time = time.perf_counter() - start_time
    return None, nodes_visited, max_memory, elapsed_time


# Initialize the start state
start_state = [-1, -1, -1, 0, 1, 1, 1]

# Run the BFS algorithm
bfs_solution, bfs_nodes_visited, bfs_max_memory, bfs_time = bfs(start_state)

# Run the DFS algorithm
dfs_solution, dfs_nodes_visited, dfs_max_memory, dfs_time = dfs(start_state)

# Compare solutions and complexities
print("Breadth-First Search (BFS):")
if bfs_solution:
    print("Solution found with", len(bfs_solution) - 1, "steps.")
    print("Solution Path:")
    for step in bfs_solution:
        print(step)
else:
    print("No solution found.")
print(f"Nodes visited: {bfs_nodes_visited}")
print(f"Maximum memory usage: {bfs_max_memory}")
print(f"Time taken: {bfs_time:.10f} seconds")  # Increase decimal places for precision

print("\nDepth-First Search (DFS):")
if dfs_solution:
    print("Solution found with", len(dfs_solution) - 1, "steps.")
    print("Solution Path:")
    for step in dfs_solution:
        print(step)
else:
    print("No solution found.")
print(f"Nodes visited: {dfs_nodes_visited}")
print(f"Maximum memory usage: {dfs_max_memory}")
print(f"Time taken: {dfs_time:.10f} seconds")  # Increase decimal places for precision
