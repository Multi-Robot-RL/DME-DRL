import torch

def generate_random_free_location(ground_truth_obstacle_map):
    """
    Generate a random location within the map that is not an obstacle.

    Args:
        ground_truth_obstacle_map (np.ndarray): Ground truth obstacle map (1 = obstacle, 0 = free space).

    Returns:
        tuple: (x, y) coordinates of the random free location.
    """
    # Get indices of all free cells (value 0)
    # free_space_indices = np.argwhere(ground_truth_obstacle_map == 0)
    #
    # if free_space_indices.size == 0:
    #     raise ValueError("No free space available in the obstacle map.")

    # Randomly select one of the free cells
    # random_index = np.random.choice(len(free_space_indices))
    # random_free_location = tuple(free_space_indices[random_index])

    # return random_free_location
    return (200, 200)


# Environment setup
def create_environment(houseexpo_dataset, room_id):
    """
    Create an environment for a specific room using the HouseExpo dataset.

    Args:
        houseexpo_dataset (Dataset): The loaded HouseExpo dataset.
        room_id (int): The ID of the room to initialize.

    Returns:
        tuple: (ground_truth_obstacle_map, frontier_map, robot_obstacle_map).
    """
    # Load the specific room from the dataset
    ground_truth_obstacle_map = torch.tensor(houseexpo_dataset[room_id]["binary_mask"])
    print(f"Map size of Room {room_id}: {ground_truth_obstacle_map.shape}")

    # Initialize the frontier map and robot obstacle map
    map_size = ground_truth_obstacle_map.shape
    frontier_map = torch.zeros_like(ground_truth_obstacle_map)  # All cells unexplored initially
    robot_obstacle_map = torch.zeros_like(
            ground_truth_obstacle_map
    )  # Robot has no knowledge of obstacles initially

    return ground_truth_obstacle_map, frontier_map, robot_obstacle_map


