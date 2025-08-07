"""
This module provides an implementation of Potential-Based Intrinsic Motivation (PBIM)
as a wrapper around the ICM_PPO algorithm.

The implementation is based on the paper "Potential-Based Reward Shaping For Intrinsic
Motivation" by Forbes et al. (2024), which can be found at:
https://arxiv.org/pdf/2402.07411
"""

from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict
from torch.nn.functional import mse_loss

from agilerl.algorithms.icm_ppo import ICM_PPO
from agilerl.components.icm import ICM


class RunningMeanStd:
    """
    Computes the running mean and standard deviation of a data stream.

    This class is used for normalization, as described in the PBIM paper.
    It maintains a running count, mean, and variance, which are updated
    incrementally with each new batch of data.

    :param shape: The shape of the data being normalized.
    :type shape: Tuple[int, ...]
    :param device: The device to store the tensors on.
    :type device: str
    """

    def __init__(self, shape: Tuple[int, ...] = (), device: str = "cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        """
        Updates the running mean and variance with a new batch of data.

        :param x: The new batch of data.
        :type x: torch.Tensor
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class PBIM_ICM(ICM):
    """
    An ICM module extension that supports PBIM by providing access to the
    forward model's prediction of the next state embedding.

    This class inherits from the standard ICM module and overrides the
    `compute_loss` method to return the predicted next state embedding,
    which is used as the potential function in PBIM.

    :param args: Positional arguments to pass to the ICM constructor.
    :param kwargs: Keyword arguments to pass to the ICM constructor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, *args, **kwargs
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,  # Add predicted_phi_next_state to return tuple
    ]:
        """
        Computes the ICM loss and returns it along with intermediate values,
        including the predicted next state embedding from the forward model.

        :param args: Positional arguments passed to the parent's compute_loss.
        :param kwargs: Keyword arguments passed to the parent's compute_loss.
        :return: A tuple containing losses, hidden states, and the predicted next state.
        """
        (
            phi_state,
            phi_next_state,
            hidden_state,
            hidden_state_next,
        ) = self.embed_obs(*args, **kwargs)
        action_input = kwargs.get("action_input", None)

        # Get predicted action
        pred_action = self.inverse_model(phi_state, phi_next_state)

        # Get predicted next state
        pred_phi_next_state = self.forward_model(phi_state, action_input)

        # Calculate inverse loss
        inverse_loss = mse_loss(pred_action, action_input)

        # Calculate forward loss
        forward_loss = mse_loss(pred_phi_next_state, phi_next_state)

        return (
            inverse_loss,
            forward_loss,
            phi_next_state,
            hidden_state,
            hidden_state_next,
            pred_phi_next_state,
        )


class PBIM_ICM_PPO(ICM_PPO):
    """
    An implementation of PPO with Potential-Based Intrinsic Motivation (PBIM).

    This class wraps the ICM_PPO algorithm and modifies its reward calculation
    to use potential-based shaping, which is guaranteed to not alter the
    set of optimal policies. It normalizes the intrinsic rewards and uses the
    ICM's forward model prediction as the potential function.

    :param args: Positional arguments to pass to the ICM_PPO constructor.
    :param kwargs: Keyword arguments to pass to the ICM_PPO constructor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pbim = kwargs.get("pbim", False)

        if self.pbim:
            self.reward_normalizer = RunningMeanStd(device=self.device)

    def _learn_from_rollout_buffer_flat(
        self, buffer_td_external: Optional[TensorDict] = None
    ) -> Dict[str, float]:
        """
        Learns from the rollout buffer using PBIM for non-recurrent policies.
        """
        if not self.pbim:
            return super()._learn_from_rollout_buffer_flat(buffer_td_external)

        if buffer_td_external:
            buffer_td = buffer_td_external
        else:
            buffer_td = self.rollout_buffer.get_tensor_batch(device=self.device)

        potential, next_potential = self.get_potentials(
            action_batch=buffer_td["actions"],
            obs_batch=buffer_td["observations"],
            next_obs_batch=buffer_td["next_observations"],
            embedded_obs=buffer_td["encoder_out"],
        )

        # Zero out potential for terminal states, as per PBRS for episodic tasks
        dones = buffer_td["dones"].to(self.device)
        next_potential = next_potential * (1.0 - dones)

        # Compute potential-based shaping reward F(s, s') = gamma * Phi(s') - Phi(s)
        pbim_rewards = self.gamma * next_potential - potential

        # Normalize the potential-based rewards
        self.reward_normalizer.update(pbim_rewards)
        normalized_pbim_rewards = pbim_rewards / torch.sqrt(
            self.reward_normalizer.var + 1e-8
        )

        # Combine with extrinsic rewards
        rewards = buffer_td["rewards"].to(self.device)
        combined_rewards = (
            rewards + self.intrinsic_reward_weight * normalized_pbim_rewards
        )
        buffer_td["rewards"] = combined_rewards.cpu()

        # Continue with standard PPO learning on the modified rewards
        return super()._learn_from_rollout_buffer_flat(buffer_td_external=buffer_td)

    def _learn_from_rollout_buffer_bptt(self) -> Dict[str, float]:
        """
        Learns from the rollout buffer using PBIM for recurrent policies (BPTT).
        """
        if not self.pbim:
            return super()._learn_from_rollout_buffer_bptt()

        buffer_td = self.rollout_buffer.get_tensor_batch(device=self.device)
        num_sequences = buffer_td["observations"].shape[1]

        # Reshape for sequence-based processing
        obs = buffer_td["observations"].reshape(
            -1, *buffer_td["observations"].shape[2:]
        )
        next_obs = buffer_td["next_observations"].reshape(
            -1, *buffer_td["next_observations"].shape[2:]
        )
        actions = buffer_td["actions"].reshape(-1, *buffer_td["actions"].shape[2:])
        encoder_out = buffer_td["encoder_out"].reshape(
            -1, *buffer_td["encoder_out"].shape[2:]
        )
        rewards = buffer_td["rewards"].to(
            self.device
        )  # Keep shape (num_steps, num_envs, 1)

        potential, next_potential = self.get_potentials(
            action_batch=actions,
            obs_batch=obs,
            next_obs_batch=next_obs,
            embedded_obs=encoder_out,
        )

        # Reshape potentials and dones back to sequence form
        potential = potential.reshape(-1, num_sequences, 1)
        next_potential = next_potential.reshape(-1, num_sequences, 1)
        dones = buffer_td["dones"].to(
            self.device
        )  # shape is (num_steps, num_envs, 1)

        # Zero out potential for terminal states
        next_potential = next_potential * (1.0 - dones)

        # Compute potential-based shaping reward F(s, s') = gamma * Phi(s') - Phi(s)
        pbim_rewards = self.gamma * next_potential - potential

        # Normalize the potential-based rewards
        # Flatten for normalizer update, then reshape back
        self.reward_normalizer.update(pbim_rewards.reshape(-1, 1))
        normalized_pbim_rewards = pbim_rewards / torch.sqrt(
            self.reward_normalizer.var + 1e-8
        )

        # Combine with extrinsic rewards
        combined_rewards = (
            rewards + self.intrinsic_reward_weight * normalized_pbim_rewards
        )
        buffer_td["rewards"] = combined_rewards.cpu()

        # Let the parent class handle the rest of the BPTT update
        return super()._learn_from_rollout_buffer_bptt()

    def get_potentials(
        self,
        action_batch: Any,
        obs_batch: Optional[Any] = None,
        next_obs_batch: Optional[Any] = None,
        embedded_obs: Optional[torch.Tensor] = None,
        embedded_next_obs: Optional[torch.Tensor] = None,
        hidden_state_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_state_next_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the potential values for PBIM.

        This method uses the ICM's forward dynamics model to get the predicted
        and actual next state embeddings, which serve as the potentials.

        :return: A tuple containing the current potential Phi(s,a) and next potential Phi(s').
        """
        obs_batch_t, action_batch_t, next_obs_batch_t = self.to_device(
            obs_batch, action_batch, next_obs_batch
        )

        if self.recurrent:
            action_input = action_batch_t
        else:
            action_input = self.icm.actions_to_one_hot(
                action_batch_t, self.icm.action_space
            )

        (
            _,  # inverse_loss
            _,  # forward_loss
            phi_next_state,
            _,  # icm_hidden_state
            _,  # icm_next_hidden_state
            pred_phi_next_state,  # This is used for our potential function Phi
        ) = self.icm.compute_loss(
            obs_batch_t=obs_batch_t,
            action_batch_t=action_batch_t,
            next_obs_batch_t=next_obs_batch_t,
            action_input=action_input,
            embedded_obs=embedded_obs,
            embedded_next_obs=embedded_next_obs,
            hidden_state=hidden_state_obs,
            hidden_state_next=hidden_state_next_obs,
        )

        # The potential Phi(s,a) is derived from the predicted next state embedding
        potential = pred_phi_next_state.mean(dim=-1, keepdim=True)
        # The next potential Phi(s') is derived from the actual next state embedding
        next_potential = phi_next_state.mean(dim=-1, keepdim=True)

        return potential, next_potential
