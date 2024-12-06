import torch.nn as nn
import torch.nn.functional as F
from hyperparam import ROOM_SPLIT, HIDDEN_DIM, OUTPUT_DIM


class ActorCriticDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the Actor-Critic model.
        Args:
            input_dim (int): The dimension of the input feature vector.
            hidden_dim (int): The number of units in the hidden layers.
            output_dim (int): The number of action outputs (e.g., angular and linear velocity).
        """
        super(ActorCriticDQN, self).__init__()

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
        )  # Outputs angular and linear velocities

        # Critic branch
        critic_hidden = F.relu(self.critic_fc1(x))
        critic_output = self.critic_output(critic_hidden)  # Outputs state-action value

        return actor_output, critic_output


# Parameters for the model
input_dim = 4 * ROOM_SPLIT  # 4 features per section, multiplied by ROOM_SPLIT
# Create the Actor-Critic model
model = ActorCriticDQN(
    input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
)

# Print the model architecture
print(model)
