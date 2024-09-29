import heapq
import numpy as np


# Levenshtein distance function
def levenshtein_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    return dp[len1][len2]


# A* search algorithm for text alignment
def a_star_search(doc1, doc2):
    start_state = (0, 0, 0)  # (index_doc1, index_doc2, accumulated_cost)
    goal_state = (len(doc1), len(doc2))
    priority_queue = [(0, start_state)]  # (priority, state)

    visited = set()

    while priority_queue:
        current_priority, current_state = heapq.heappop(priority_queue)
        idx1, idx2, cost = current_state

        if (idx1, idx2) in visited:
            continue

        visited.add((idx1, idx2))

        # Check if goal state is reached
        if idx1 == goal_state[0] and idx2 == goal_state[1]:
            return cost

        # Generate possible transitions and their costs
        if idx1 < len(doc1) and idx2 < len(doc2):
            new_cost = cost + levenshtein_distance(doc1[idx1], doc2[idx2])
            heapq.heappush(priority_queue, (new_cost, (idx1 + 1, idx2 + 1, new_cost)))

        # Skip sentence in doc1
        if idx1 < len(doc1):
            new_cost = cost + len(doc1[idx1])  # Maximum cost for skipping
            heapq.heappush(priority_queue, (new_cost, (idx1 + 1, idx2, new_cost)))

        # Skip sentence in doc2
        if idx2 < len(doc2):
            new_cost = cost + len(doc2[idx2])  # Maximum cost for skipping
            heapq.heappush(priority_queue, (new_cost, (idx1, idx2 + 1, new_cost)))


# Preprocessing the text by tokenizing into sentences and normalizing
def preprocess_document(doc):
    sentences = doc.split('.')
    sentences = [s.strip().lower() for s in sentences if s.strip()]
    return sentences


# Test case runner
def run_test_case(doc1, doc2, test_case_number):
    doc1_sentences = preprocess_document(doc1)
    doc2_sentences = preprocess_document(doc2)

    alignment_cost = a_star_search(doc1_sentences, doc2_sentences)

    print(f"Test Case {test_case_number}:")
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Alignment Cost: {alignment_cost}")
    print("\n" + "=" * 40 + "\n")


# Test Cases

# Test Case 1: Identical Documents
doc1 = "This is a test document. It contains several sentences."
doc2 = "This is a test document. It contains several sentences."
run_test_case(doc1, doc2, 1)

# Test Case 2: Slightly Modified Document
doc1 = "This is a test document. It contains several sentences."
doc2 = "This is a sample document. It includes several sentences."
run_test_case(doc1, doc2, 2)

# Test Case 3: Completely Different Documents
doc1 = "This is a test document. It contains several sentences."
doc2 = "The quick brown fox jumps over the lazy dog."
run_test_case(doc1, doc2, 3)

# Test Case 4: Partial Overlap
doc1 = "This is a test document. It contains several sentences. This part is unique to doc1."
doc2 = "This is a test document. It contains some sentences. This part is unique to doc2."
run_test_case(doc1, doc2, 4)
