import torch
import math
from hyperparam import MAX_EPISODES, TRAIN_DATASET_SIZE
from utils import generate_random_free_location, create_environment
from dataset import load_processed_dataset

def calculate_potentially_explorable_distance_batched(
    current_locations, current_directions, rooms, frontier_maps, max_detection_dist, num_directions=16
):
    """
    Batched calculation of potentially explorable distances for multiple rooms.

    Args:
        current_locations (torch.Tensor): Current (x, y) coordinates of robots (batch_size, 2).
        current_directions (torch.Tensor): Current directions of robots in radians (batch_size,).
        rooms (torch.Tensor): Room tensors where 1 = obstacle, 0 = free space (batch_size, H, W).
        frontier_maps (torch.Tensor): Maps indicating explored areas (batch_size, H, W).
        max_detection_dist (float): Maximum detection range.
        num_directions (int): Number of angular sections.

    Returns:
        torch.Tensor: Potentially explorable distances for each direction (batch_size, num_directions).
    """
    batch_size, height, width = rooms.shape
    device = rooms.device

    # Generate angles for ray directions
    angles = torch.linspace(0, 2 * math.pi, num_directions, device=device)  # (num_directions,)
    angles = angles.unsqueeze(0) + current_directions.unsqueeze(1)  # (batch_size, num_directions)

    # Compute step offsets for all rays
    dx = torch.cos(angles)  # (batch_size, num_directions)
    dy = torch.sin(angles)  # (batch_size, num_directions)

    # Steps along the ray
    steps = torch.linspace(0, max_detection_dist, steps=100, device=device)  # (num_steps,)
    dx_steps = dx.unsqueeze(2) * steps  # (batch_size, num_directions, num_steps)
    dy_steps = dy.unsqueeze(2) * steps  # (batch_size, num_directions, num_steps)

    # Compute candidate positions along rays
    candidate_x = current_locations[:, 0].unsqueeze(1).unsqueeze(2) + dx_steps  # (batch_size, num_directions, num_steps)
    candidate_y = current_locations[:, 1].unsqueeze(1).unsqueeze(2) + dy_steps  # (batch_size, num_directions, num_steps)

    # Convert positions to integers
    candidate_x = candidate_x.long().clamp(0, height - 1)
    candidate_y = candidate_y.long().clamp(0, width - 1)

    # Initialize explorable distances
    explorable_distances = torch.full((batch_size, num_directions), max_detection_dist, device=device)

    # Gather room and frontier map values
    for step_idx in range(steps.size(0)):
        x_indices = candidate_x[..., step_idx]
        y_indices = candidate_y[..., step_idx]

        room_values = rooms[torch.arange(batch_size).unsqueeze(1), x_indices, y_indices]
        frontier_values = frontier_maps[torch.arange(batch_size).unsqueeze(1), x_indices, y_indices]

        # Identify collisions
        collisions = (room_values == 1) | (frontier_values == 1)

        # Update distances where collisions occur for the first time
        collision_mask = (explorable_distances == max_detection_dist) & collisions
        explorable_distances[collision_mask] = steps[step_idx]

        # Stop processing further steps where collisions have already occurred
        if collision_mask.all():
            break
    print(explorable_distances)
    return explorable_distances


def explore_rooms_batched(
    rooms,
    current_locations,
    current_directions,
    max_detection_dist=10.0,
    max_steps=1000,
    num_directions=16,
):
    """
    Batched exploration of multiple rooms using a systematic strategy.

    Args:
        rooms (torch.Tensor): Room tensors where 1 = obstacle, 0 = free space (batch_size, H, W).
        current_locations (torch.Tensor): Initial (x, y) locations of robots (batch_size, 2).
        current_directions (torch.Tensor): Initial directions of robots in radians (batch_size,).
        max_detection_dist (float): Maximum detection range.
        max_steps (int): Maximum steps for exploration.
        num_directions (int): Number of angular sections to consider.

    Returns:
        torch.Tensor: Final explored maps (batch_size, H, W).
        torch.Tensor: Paths taken by each robot (batch_size, max_steps, 2).
    """
    batch_size, height, width = rooms.shape
    frontier_maps = torch.zeros_like(rooms, dtype=torch.float32, device=rooms.device)
    paths = torch.zeros((batch_size, max_steps, 2), dtype=torch.float32, device=rooms.device)

    # Initialize frontier maps and paths
    for i in range(batch_size):
        frontier_maps[i, int(current_locations[i, 0]), int(current_locations[i, 1])] = 1
        paths[i, 0] = current_locations[i]

    for step in range(1, max_steps):
        # Calculate potentially explorable distances for all rooms
        explorable_distances = calculate_potentially_explorable_distance_batched(
            current_locations, current_directions, rooms, frontier_maps, max_detection_dist, num_directions
        )

        # Determine best directions for each robot
        best_directions = explorable_distances.argmax(dim=1)
        angle_increment = 2 * math.pi / num_directions
        best_angles = current_directions + best_directions * angle_increment

        # Prepare the next locations
        next_locations = current_locations.clone()
        for i in range(batch_size):
            dx = torch.cos(best_angles[i])
            dy = torch.sin(best_angles[i])
            candidate_location = (
                int(current_locations[i, 0] + dx),
                int(current_locations[i, 1] + dy),
            )

            # Check boundaries and obstacles
            if (
                0 <= candidate_location[0] < height
                and 0 <= candidate_location[1] < width
                and rooms[i, candidate_location[0], candidate_location[1]] == 0
            ):
                # Valid move
                next_locations[i] = torch.tensor(candidate_location, device=rooms.device)
                frontier_maps[i, candidate_location[0], candidate_location[1]] = 1  # Update frontier map
                current_directions[i] = best_angles[i]  # Update direction

        # Log positions in paths
        paths[:, step] = next_locations

        # Update robot states
        current_locations = next_locations

    return frontier_maps, paths


if __name__ == "__main__":
    dataset = load_processed_dataset(0, 5)  # Load a subset of the dataset (5 rooms)
    batch_size = 3  # Number of rooms to explore simultaneously
    MAX_EPISODES = 10  # Number of episodes
    MAX_DETECTION_DIST = 10.0
    MAX_STEPS = 1000
    NUM_DIRECTIONS = 16

    for episode in range(MAX_EPISODES):
        # Initialize environment and robot states for the batch
        ground_truth_obstacle_maps = []
        current_locations = []
        current_directions = []

        for i in range(batch_size):
            ground_truth_obstacle_map, frontier_map, robot_obstacle_map = create_environment(
                dataset, i % len(dataset)  # Cycle through the dataset
            )
            ground_truth_obstacle_maps.append(ground_truth_obstacle_map)
            # Generate random initial robot locations and directions
            current_location = generate_random_free_location(torch.tensor(ground_truth_obstacle_map))
            current_direction = torch.rand(1).item() * 2 * math.pi  # Random direction in [0, 2π]
            current_locations.append(current_location)
            current_directions.append(current_direction)

        # Convert lists to batched tensors
        ground_truth_obstacle_maps = torch.stack(
            [torch.tensor(map_, dtype=torch.float32) for map_ in ground_truth_obstacle_maps]
        )
        current_locations = torch.tensor(current_locations, dtype=torch.float32)
        current_directions = torch.tensor(current_directions, dtype=torch.float32)

        # Explore the rooms in batch
        frontier_maps, paths = explore_rooms_batched(
            ground_truth_obstacle_maps,
            current_locations,
            current_directions,
            max_detection_dist=MAX_DETECTION_DIST,
            max_steps=MAX_STEPS,
            num_directions=NUM_DIRECTIONS,
        )

        # Output exploration results
        print(f"Episode {episode + 1}/{MAX_EPISODES}")
        for i, path in enumerate(paths):
            print(f"Room {i}: Path length: {len(path)}")
            print(path[1]-path[0])
