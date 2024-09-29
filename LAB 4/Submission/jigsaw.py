import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and process the image
image_path = r'D:\AI LAB\photo.jpg'  # Replace with your image path
img = Image.open(image_path)
img_np = np.array(img)

# Function to divide the image into blocks (tiles)
def divide_image(image, block_size):
    tiles = []
    height, width = image.shape[:2]
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Ensure we do not exceed the image dimensions
            tile = image[i:i + block_size, j:j + block_size]
            if tile.shape[0] == block_size and tile.shape[1] == block_size:
                tiles.append(tile)
    return tiles

# Function to stitch the tiles back into a single image
def stitch_image(tiles, grid_size, block_size):
    stitched = np.zeros((block_size * grid_size, block_size * grid_size, 3), dtype=tiles[0].dtype)
    for idx, tile in enumerate(tiles):
        i, j = divmod(idx, grid_size)
        stitched[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = tile
    return stitched

# Set block size (depending on how you want to split the image)
block_size = 128  # Assuming we want to split the image into 4x4 blocks
tiles = divide_image(img_np, block_size)

# Check if we have the expected number of tiles
grid_size = int(np.sqrt(len(tiles)))
print(f"Total tiles created: {len(tiles)}; Grid size: {grid_size}x{grid_size}")

# Function to compute the energy (difference) between adjacent tiles
def get_tile_difference(tile1, tile2, axis=0):
    if axis == 0:  # vertical difference
        return np.linalg.norm(tile1[-1, :, :] - tile2[0, :, :])
    else:  # horizontal difference
        return np.linalg.norm(tile1[:, -1, :] - tile2[:, 0, :])

# Calculate the energy of the current configuration
def calculate_energy(tiles, grid_size):
    energy = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1:  # Vertical neighbor
                energy += get_tile_difference(tiles[i * grid_size + j], tiles[(i + 1) * grid_size + j], axis=0)
            if j < grid_size - 1:  # Horizontal neighbor
                energy += get_tile_difference(tiles[i * grid_size + j], tiles[i * grid_size + j + 1], axis=1)
    return energy

# Simulated annealing algorithm
def simulated_annealing(tiles, grid_size, max_iter=10000, temp=1000, decay=0.9995):
    best = tiles.copy()  # Initial configuration
    best_energy = calculate_energy(best, grid_size)
    current = best.copy()
    current_energy = best_energy

    for i in range(max_iter):
        new_tiles = current.copy()
        # Swap two random tiles
        idx1, idx2 = np.random.choice(len(new_tiles), size=2, replace=False)
        new_tiles[idx1], new_tiles[idx2] = new_tiles[idx2], new_tiles[idx1]

        new_energy = calculate_energy(new_tiles, grid_size)

        # Check if the new configuration is better
        if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy) / temp):
            current = new_tiles
            current_energy = new_energy
            if new_energy < best_energy:
                best = new_tiles
                best_energy = new_energy

        # Decay the temperature
        temp *= decay

        # Display progress
        if i % 1000 == 0:
            print(f"Iteration {i}, Energy: {best_energy}")
            reconstructed_img = stitch_image(best, grid_size, block_size)
            plt.imshow(reconstructed_img)
            plt.title(f'Iteration {i}, Energy: {best_energy}')
            plt.axis('off')
            plt.show()

    return best

# Apply simulated annealing to solve the puzzle
solved_tiles = simulated_annealing(tiles, grid_size)

# Stitch the solved tiles back into an image
solved_image = stitch_image(solved_tiles, grid_size, block_size)

# Show the final solved image
plt.imshow(solved_image)
plt.title('Solved Image')
plt.axis('off')
plt.show()
