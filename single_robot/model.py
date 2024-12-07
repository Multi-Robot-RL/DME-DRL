import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparam import (
    MAX_ANGULAR_VELOCITY,
    MAX_LINEAR_VELOCITY,
    ROOM_SPLIT,
    HIDDEN_DIM,
    OUTPUT_DIM,
)


class ActorCriticDQN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        max_linear_velocity,
        max_angular_velocity,
    ):
        """
        Initialize the Actor-Critic model with velocity restrictions.
        Args:
            input_dim (int): The dimension of the input feature vector.
            hidden_dim (int): The number of units in the hidden layers.
            output_dim (int): The number of action outputs (e.g., angular and linear velocity).
            max_linear_velocity (float): Maximum linear velocity.
            max_angular_velocity (float): Maximum angular velocity.
        """
        super(ActorCriticDQN, self).__init__()

        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        # Shared layers for feature extraction
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor network
        self.actor_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_output = nn.Linear(hidden_dim, output_dim)

        # Critic network
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input feature tensor.
        Returns:
            torch.Tensor: Actor output (angular and linear velocity).
            torch.Tensor: Critic output (state value).
        """
        # Shared feature extraction
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # Actor branch
        actor_hidden = F.relu(self.actor_fc1(x))
        actor_output = self.actor_output(
            actor_hidden
        )  # Outputs raw angular and linear velocities

        # Restrict velocities
        linear_velocity = torch.tanh(actor_output[:, 0]) * self.max_linear_velocity
        angular_velocity = torch.tanh(actor_output[:, 1]) * self.max_angular_velocity
        restricted_actor_output = torch.stack(
            [linear_velocity, angular_velocity], dim=1
        )

        # Critic branch
        critic_hidden = F.relu(self.critic_fc1(x))
        critic_output = self.critic_output(critic_hidden)  # Outputs state-action value

        return restricted_actor_output, critic_output


# Parameters for the model
input_dim = 4 * ROOM_SPLIT + 5  # 4 features per section, multiplied by ROOM_SPLIT and the robot location, room size and direction
# Create the Actor-Critic model
model = ActorCriticDQN(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    max_linear_velocity=MAX_LINEAR_VELOCITY,
    max_angular_velocity=MAX_ANGULAR_VELOCITY,
)

# Print the model architecture
print(model)
