import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Step 1: Define Rajasthan Tourist Locations and Coordinates
locations = {
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Jaisalmer": (26.9157, 70.9160),
    "Pushkar": (26.4922, 74.5494),
    "Ajmer": (26.4530, 74.6399),
    "Bikaner": (28.0229, 73.3119),
    "Mount Abu": (24.5926, 72.7022),
    "Chittorgarh": (24.8793, 74.6310),
    "Ranthambore": (26.0173, 76.6342),
    "Kota": (25.2138, 75.8648),
    "Bundi": (25.4472, 75.6367),
    "Shekhawati": (28.3852, 75.2597),
    "Mandawa": (28.3568, 75.1340),
    "Neemrana": (27.9165, 76.5615),
    "Sawai Madhopur": (25.9910, 76.3910),
    "Sikar": (27.6096, 75.1392),
    "Barmer": (25.7575, 71.4285),
    "Jhalawar": (24.5890, 76.1688),
    "Tonk": (26.0469, 75.6165),
}


# Step 2: Create Distance Matrix
def create_distance_matrix(locations):
    cities = list(locations.keys())
    n = len(cities)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = locations[cities[i]]
                lat2, lon2 = locations[cities[j]]
                distance_matrix[i][j] = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)  # Euclidean distance

    return distance_matrix, cities


# Create the distance matrix
distance_matrix, cities = create_distance_matrix(locations)


# Step 3: Implement Simulated Annealing with Fixed Start and End
def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + dist_matrix[tour[-1], tour[0]]


def simulated_annealing_fixed_start(locations, dist_matrix, start_city_index, initial_temp=1000, cooling_rate=0.995,
                                    max_iterations=1000):
    # Start with a fixed starting city
    current_tour = [start_city_index] + list(range(len(locations)))  # Include the starting city first
    current_tour.remove(start_city_index)  # Remove it from the random selection of cities
    random.shuffle(current_tour)  # Shuffle to create an initial solution
    current_tour = [start_city_index] + current_tour  # Re-add the starting city to the front

    best_tour = current_tour[:]
    best_cost = total_distance(best_tour, dist_matrix)

    temp = initial_temp  # Initial temperature

    for i in range(max_iterations):
        # Generate a neighbor by swapping two cities (excluding the first city)
        idx = random.sample(range(1, len(locations)), 2)  # Only swap cities after the first
        idx.sort()
        next_tour = current_tour[:]
        next_tour[idx[0]:idx[1] + 1] = reversed(next_tour[idx[0]:idx[1] + 1])  # Reverse segment

        next_cost = total_distance(next_tour, dist_matrix)

        # Acceptance criteria
        if next_cost < best_cost or random.random() < math.exp((best_cost - next_cost) / temp):
            current_tour = next_tour
            if next_cost < best_cost:
                best_tour = next_tour
                best_cost = next_cost

        temp *= cooling_rate  # Cool down

    return best_tour, best_cost


# Step 4: Run Simulated Annealing for Rajasthan with a Fixed Starting Point
start_city = "Jaipur"  # Example starting city
start_city_index = list(locations.keys()).index(start_city)  # Get the index of the starting city

best_tour, best_cost = simulated_annealing_fixed_start(locations, distance_matrix, start_city_index)
print(f"Best Tour: {[cities[i] for i in best_tour]}")
print(f"Best Cost: {best_cost:.2f}")


# Step 5: Visualize the Best Tour
def plot_tour(locations, tour):
    coords = np.array(
        [locations[list(locations.keys())[i]] for i in tour + [tour[0]]])  # Append the first city to close the loop
    plt.figure(figsize=(10, 6))
    plt.plot(coords[:, 1], coords[:, 0], marker='o', linestyle='-')  # Plot using longitude and latitude
    for i, city in enumerate(list(locations.keys())):
        plt.annotate(city, (coords[i][1], coords[i][0]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('Best Tour in Rajasthan')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()


# Visualize the best tour
plot_tour(locations, best_tour)
