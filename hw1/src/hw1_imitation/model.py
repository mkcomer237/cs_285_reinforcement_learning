"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = self.state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.action_dim * self.chunk_size))

        self.policy = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict action chunks."""
        y = self.policy(state)
        # print("y.shape:", y.shape)
        return y.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        prediction_chunk = self(state)
        loss_fn =nn.MSELoss()
        loss = loss_fn(prediction_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # Output the policy generated action chunk for a given state
        with torch.no_grad():
            return self(state)



class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = self.chunk_size * self.action_dim + self.state_dim + + 1 # observed state, noisy action chunk, and tau  
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.action_dim * self.chunk_size))

        self.policy = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        # Add noise? 
        print(f"State shape: {state.shape}")
        batch_size = state.shape[0]
        print(f"Batch size: {batch_size}")
        # Sample from a (0, 1) normal distribution for the full output size.  
        # This must be combined with the true label (action chunk) and matches dims with that
        tau = torch.rand(batch_size, 1, 1) # Randomly sample tau before interpolation.  Separate tau for each batch item.  
        noise = torch.randn(batch_size, self.chunk_size, self.action_dim) # mean 0, std 1
        print(f"Noise shape: {noise.shape}")
        print(f"Tau shape: {tau.shape}")
        print(f"Action chunk shape: {action_chunk.shape}")
        interpolated_action_chunk = (1 - tau) * action_chunk + tau * noise
        # Adding noise generates the training sample.  The original sample without noise is the target.
        # Our policy should output a result with the same dimensiopns as the training sample At and noise
        # Policy needs to take in tau, the interpolated action chunk -> noise, and the observed state Ot
        reshaped_interpolated_action_chunk = interpolated_action_chunk.reshape(-1, self.action_dim*self.chunk_size)
        print(f"Reshaped interpolated action chunk shape: {reshaped_interpolated_action_chunk.shape}")
        X = torch.cat([reshaped_interpolated_action_chunk, state, tau.reshape(-1, 1)], dim=1)
        print("X shape: ", X.shape)
        y = self.policy(X)
        print(f"y shape: {y.shape}")

        return y.view(-1, self.chunk_size, self.action_dim)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # This is the inference step, we need to use the Euler integration to sequentially go through values of tau to get a trajectory towards a final denoised action chunk.
        raise NotImplementedError


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
