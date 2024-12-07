import numpy as np


def evaluate_model_performance(frontier_map, ground_truth_obstacle_map):
    """
    Evaluate the performance of a model by calculating the percentage of free space discovered.

    Args:
        frontier_map (np.ndarray): The frontier map indicating explored areas (0 = unexplored, 1 = explored).
        ground_truth_obstacle_map (np.ndarray): The ground truth obstacle map (1 = obstacle, 0 = free space).

    Returns:
        float: Percentage of free space marked as discovered.
    """
    # Identify free space in the ground truth obstacle map
    free_space_mask = ground_truth_obstacle_map == 0

    # Count total free space cells
    total_free_space = np.sum(free_space_mask)

    # Identify discovered free space in the frontier map
    discovered_free_space = np.sum(free_space_mask & (frontier_map == 1))

    # Calculate percentage of free space discovered
    if total_free_space == 0:
        return 0.0  # Handle edge case where no free space exists
    percentage_discovered = (discovered_free_space / total_free_space) * 100

    return percentage_discovered
