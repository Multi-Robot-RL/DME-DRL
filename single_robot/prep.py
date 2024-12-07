import torch
import torch.optim as optim
import torch.nn.functional as F
from model import device
# Generate random free location
def generate_random_free_location(ground_truth_obstacle_map):
    """
    Generate a random location within the map that is not an obstacle.

    Args:
        ground_truth_obstacle_map (torch.Tensor): Ground truth obstacle map (1 = obstacle, 0 = free space).

    Returns:
        tuple: (x, y) coordinates of the random free location.
    """
    # Find free space indices
    free_space_indices = torch.nonzero(ground_truth_obstacle_map == 0, as_tuple=False)

    # Check if free space exists
    if free_space_indices.size(0) == 0:
        raise ValueError("No free space available in the obstacle map.")

    # Randomly select one of the free cells
    random_index = torch.randint(0, free_space_indices.size(0), (1,)).item()
    random_free_location = tuple(free_space_indices[random_index].tolist())

    return random_free_location


# Environment setup
def create_environment(houseexpo_dataset, room_id):
    ground_truth_obstacle_map = torch.tensor(
        houseexpo_dataset[room_id]["binary_mask"], device=device
    )
    print(f"Map size of Room {room_id}: {ground_truth_obstacle_map.shape}")
    map_size = ground_truth_obstacle_map.shape
    frontier_map = torch.zeros(map_size, device=device)
    robot_obstacle_map = torch.zeros(map_size, device=device)
    return ground_truth_obstacle_map, frontier_map, robot_obstacle_map

