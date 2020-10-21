import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=-1)
        )

        self.log_probs = []
        self.rewards = []

    def forward(self, input):
        return self.mlp_layer(input)

    def act(self, input):
        prob = self.forward(input)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.detach().item()