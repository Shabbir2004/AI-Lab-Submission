import random

def generate_random_3sat(n, m):
    clauses = set()
    while len(clauses) < m:
        clause = set()
        while len(clause) < 3:
            var = random.randint(1, n)
            negated = random.choice([True, False])
            if negated:
                var = -var
            clause.add(var)
        clauses.add(tuple(clause))
    return list(clauses)

def heuristic1(assignment, clauses):
    # Count the number of satisfied clauses
    return sum(1 for clause in clauses if any((var > 0 and assignment[abs(var)] == 1) or (var < 0 and assignment[abs(var)] == 0) for var in clause))

def heuristic2(assignment, clauses):
    # Count the number of satisfied clauses but gives weight to the number of variables in clauses
    return sum(sum(1 for var in clause if (var > 0 and assignment[abs(var)] == 1) or (var < 0 and assignment[abs(var)] == 0)) for clause in clauses)

def hill_climbing(clauses, n, heuristic):
    assignment = [random.randint(0, 1) for _ in range(n + 1)]
    current_score = heuristic(assignment, clauses)

    while True:
        neighbors = []
        for i in range(1, n + 1):
            neighbor = assignment[:]
            neighbor[i] = 1 - neighbor[i]  # Flip the variable
            neighbors.append(neighbor)

        best_neighbor = max(neighbors, key=lambda x: heuristic(x, clauses))
        best_score = heuristic(best_neighbor, clauses)

        if best_score <= current_score:
            break
        else:
            assignment = best_neighbor
            current_score = best_score

    return assignment, current_score

def compare_hill_climbing(n, m):
    clauses = generate_random_3sat(n, m)
    solution1, score1 = hill_climbing(clauses, n, heuristic1)
    solution2, score2 = hill_climbing(clauses, n, heuristic2)
    print(f"Hill Climbing: Score with Heuristic 1: {score1}, Score with Heuristic 2: {score2}")

# Example usage for Hill Climbing
if __name__ == "__main__":
    n = 5  # Number of variables
    m = 10  # Number of clauses
    compare_hill_climbing(n, m)
