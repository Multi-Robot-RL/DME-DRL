import torch
from hyperparam import ROOM_SPLIT
from model import device


def generate_feature_vector(
    frontier_map,
    obstacle_map,
    robot_position,
    robot_direction,
    frontier_threshold,
    obstacle_threshold,
):
    """
    Generate the feature vector from frontier and obstacle maps using PyTorch (CUDA-enabled).

    Args:
        frontier_map (torch.Tensor): 2D tensor representing explored areas.
        obstacle_map (torch.Tensor): 2D tensor representing obstacles.
        robot_position (tuple): Tuple of (x, y) coordinates representing the robot's position.
        robot_direction (float): Robot's current direction in radians.
        ROOM_SPLIT (int): Number of angular sections around the robot.
        frontier_threshold (float): Threshold above which an area is considered explored.
        obstacle_threshold (float): Threshold above which a cell is considered an obstacle.
        device (str): Device to run the function on ("cuda" or "cpu").

    Returns:
        torch.Tensor: Flattened feature vector for the ROOM_SPLIT sections.
    """
    # Move inputs to device
    frontier_map = frontier_map.to(device)
    obstacle_map = obstacle_map.to(device)

    # Get map dimensions
    map_size_x, map_size_y = frontier_map.shape

    # Robot position
    robot_x, robot_y = robot_position

    # Create grid coordinates
    x_coords, y_coords = torch.meshgrid(
        torch.arange(map_size_x, device=device),
        torch.arange(map_size_y, device=device),
        indexing="ij",
    )

    # Calculate relative positions to the robot
    dx = x_coords - robot_x
    dy = y_coords - robot_y
    distances = torch.sqrt(dx**2 + dy**2)

    # Calculate angles relative to the robot
    angles = torch.atan2(dy, dx)
    angles[angles < 0] += 2 * torch.pi  # Normalize to [0, 2π)

    # Mask for the robot's current position (avoid self-reference)
    mask_robot_position = (dx == 0) & (dy == 0)

    # Mask out-of-bound areas
    mask_valid = distances > 0

    # Combine valid mask
    valid_mask = mask_valid & ~mask_robot_position

    # Angular increments for each section
    angle_increment = 2 * torch.pi / ROOM_SPLIT
    start_angles = torch.arange(ROOM_SPLIT, device=device) * angle_increment
    end_angles = start_angles + angle_increment

    # Preallocate metrics
    total_area = torch.zeros(ROOM_SPLIT, device=device)
    unexplored_area = torch.zeros(ROOM_SPLIT, device=device)
    nearest_obstacle_distance = torch.full((ROOM_SPLIT,), float("inf"), device=device)

    # Compute metrics for each angular section
    for section, (start_angle, end_angle) in enumerate(zip(start_angles, end_angles)):
        angle_mask = (angles >= start_angle) & (angles < end_angle)
        combined_mask = valid_mask & angle_mask

        # Total area in this section
        total_area[section] = combined_mask.sum()

        # Unexplored area
        unexplored_area[section] = (
            (frontier_map < frontier_threshold) & combined_mask
        ).sum()

        # Nearest obstacle distance
        obstacle_distances = distances[
            (obstacle_map > obstacle_threshold) & combined_mask
        ]
        if obstacle_distances.numel() > 0:
            nearest_obstacle_distance[section] = obstacle_distances.min()

    # Handle sections with no obstacles
    nearest_obstacle_distance[nearest_obstacle_distance == float("inf")] = (
        0.1  # Add epsilon
    )

    # Calculate unexplored ratio
    unexplored_ratio = unexplored_area / total_area
    unexplored_ratio[total_area == 0] = 0  # Avoid division by zero

    # Flatten the feature vector
    feature_vector = torch.cat(
        [
            total_area,
            1 / nearest_obstacle_distance,
            unexplored_area,
            unexplored_ratio,
            torch.tensor(robot_position, device=device),
            torch.tensor(frontier_map.shape, device=device, dtype=torch.float32),
            torch.tensor([robot_direction], device=device),
        ]
    )

    return feature_vector
