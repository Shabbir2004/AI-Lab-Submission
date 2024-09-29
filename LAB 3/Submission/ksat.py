import random

def generate_random_k_sat(n, m, k):

    clauses = set()

    while len(clauses) < m:
        clause = set()

        while len(clause) < k:
            var = random.randint(1, n)  # Randomly choose a variable
            negated = random.choice([True, False])  # Randomly choose negation
            if negated:
                var = -var
            clause.add(var)

        # Add the clause only if it has distinct variables
        if len(clause) == k:
            clauses.add(tuple(sorted(clause)))  # Sort for consistency

    return list(clauses)

def print_k_sat_problem(clauses):
    """Print the generated k-SAT problem."""
    for clause in clauses:
        print(" OR ".join(f"x{abs(var)}" if var > 0 else f"¬x{abs(var)}" for var in clause))

# Example usage
if __name__ == "__main__":
    # Accept input for k, m, and n
    k = int(input("Enter the clause length (k): "))
    m = int(input("Enter the number of clauses (m): "))
    n = int(input("Enter the number of variables (n): "))

    # Generate and print the k-SAT problem
    k_sat_problem = generate_random_k_sat(n, m, k)
    print(f"\nGenerated {m} clauses for {n} variables in a {k}-SAT problem:")
    print_k_sat_problem(k_sat_problem)
