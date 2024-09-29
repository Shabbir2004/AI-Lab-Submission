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
    return sum(1 for clause in clauses if any(
        (var > 0 and assignment[abs(var)] == 1) or (var < 0 and assignment[abs(var)] == 0) for var in clause))


def heuristic2(assignment, clauses):
    return sum(
        sum(1 for var in clause if (var > 0 and assignment[abs(var)] == 1) or (var < 0 and assignment[abs(var)] == 0))
        for clause in clauses)


def beam_search(clauses, n, beam_width, heuristic):
    initial_population = [[random.randint(0, 1) for _ in range(n + 1)] for _ in range(beam_width)]

    while True:
        scored_population = [(assignment, heuristic(assignment, clauses)) for assignment in initial_population]
        scored_population.sort(key=lambda x: x[1], reverse=True)

        # If the best score equals the number of clauses, we found a solution
        if scored_population[0][1] == len(clauses):
            return scored_population[0]

        # Select the best 'beam_width' assignments
        selected = scored_population[:beam_width]
        next_population = []
        for assignment, _ in selected:
            for i in range(1, n + 1):
                neighbor = assignment[:]
                neighbor[i] = 1 - neighbor[i]  # Flip the variable
                next_population.append(neighbor)

        initial_population = next_population


def compare_beam_search(n, m, beam_width):
    clauses = generate_random_3sat(n, m)
    solution1 = beam_search(clauses, n, beam_width, heuristic1)
    print(f"Beam Search (width={beam_width}): Score with Heuristic 1: {solution1[1]}")

    solution2 = beam_search(clauses, n, beam_width, heuristic2)
    print(f"Beam Search (width={beam_width}): Score with Heuristic 2: {solution2[1]}")


# Example usage for Beam Search
if __name__ == "__main__":
    n = 5  # Number of variables
    m = 10  # Number of clauses
    compare_beam_search(n, m, beam_width=3)
    compare_beam_search(n, m, beam_width=4)
