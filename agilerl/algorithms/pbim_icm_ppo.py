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
from gymnasium import spaces

from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.algorithms.icm_ppo import ICM_PPO
from agilerl.components.icm import ICM
from agilerl.typing import BPTTSequenceType


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

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

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
        action_input = kwargs.pop("action_input", None)
        action_batch_t = kwargs.pop("action_batch_t", None)
        (
            phi_state,
            phi_next_state,
            hidden_state,
            hidden_state_next,
        ) = self.embed_obs(*args, **kwargs)

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

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        index: int = 0,
        hp_config: Optional[HyperparameterConfig] = None,
        net_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        lr: float = 1e-4,
        learn_step: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        mut: Optional[str] = None,
        action_std_init: float = 0.0,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_images: bool = True,
        update_epochs: int = 4,
        actor_network: Optional[Any] = None,  # EvolvableModule
        critic_network: Optional[Any] = None,  # EvolvableModule
        share_encoders: bool = True,
        num_envs: int = 1,
        use_rollout_buffer: bool = True,  # Must be True for ICM_PPO current design
        rollout_buffer_config: Optional[Dict[str, Any]] = {},
        recurrent: bool = False,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True,
        torch_compiler: Optional[Any] = None,
        bptt_sequence_type: BPTTSequenceType = BPTTSequenceType.FIFTY_PERCENT_OVERLAP,
        # ICM specific parameters
        icm_lr: float = 1e-4,
        icm_beta: float = 0.2,
        intrinsic_reward_weight: float = 0.1,  # eta in the paper
        use_shared_encoder_for_icm: bool = False,
        icm_encoder_net_config: Optional[Dict[str, Any]] = None,
        icm_inverse_net_config: Optional[Dict[str, Any]] = None,
        icm_forward_net_config: Optional[Dict[str, Any]] = None,
        icm_loss_weight: float = 0.1,
        pbim: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            index=index,
            hp_config=hp_config,
            net_config=net_config,
            batch_size=batch_size,
            lr=lr,
            learn_step=learn_step,
            gamma=gamma,
            gae_lambda=gae_lambda,
            mut=mut,
            action_std_init=action_std_init,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            normalize_images=normalize_images,
            update_epochs=update_epochs,
            actor_network=actor_network,
            critic_network=critic_network,
            share_encoders=share_encoders,
            num_envs=num_envs,
            use_rollout_buffer=use_rollout_buffer,
            rollout_buffer_config=rollout_buffer_config,
            recurrent=recurrent,
            device=device,
            accelerator=accelerator,
            wrap=wrap,
            torch_compiler=torch_compiler,
            bptt_sequence_type=bptt_sequence_type,
            icm_lr=icm_lr,
            icm_beta=icm_beta,
            intrinsic_reward_weight=intrinsic_reward_weight,
            use_shared_encoder_for_icm=use_shared_encoder_for_icm,
            icm_encoder_net_config=icm_encoder_net_config,
            icm_inverse_net_config=icm_inverse_net_config,
            icm_forward_net_config=icm_forward_net_config,
            icm_loss_weight=icm_loss_weight,
            pbim=pbim,
        )

        self.pbim = pbim

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
            buffer_td = buffer_td.view(-1)

        if self.use_shared_encoder_for_icm:
            # Build (t, t+1) pairs along time, flatten over B*(T-1)
            obs_t = buffer_td["observations"][:-1]
            obs_tp = buffer_td["next_observations"][1:]
            act_t = buffer_td["actions"][:-1]

            emb_t = buffer_td["encoder_out"][:-1]
            emb_tp = buffer_td["encoder_out"][1:]
        else:
            obs_t = buffer_td["observations"]
            obs_tp = buffer_td["next_observations"]
            act_t = buffer_td["actions"]
            emb_t = buffer_td["encoder_out"]
            emb_tp = buffer_td["encoder_out"]


        with torch.no_grad():
            potential, next_potential = self.get_potentials(
                action_batch=act_t,
                obs_batch=obs_t,
                next_obs_batch=obs_tp,
                embedded_obs=emb_t,
                embedded_next_obs=emb_tp,
            )

        # Zero out potential for terminal states, as per PBRS for episodic tasks
        dones = buffer_td["dones"]
        next_potential = next_potential.masked_fill(dones, 0)

        # Compute potential-based shaping reward F(s, s') = gamma * Phi(s') - Phi(s)
        assert potential.shape == next_potential.shape, f"Potential shape: {potential.shape}, Next potential shape: {next_potential.shape}"
        pbim_rewards = self.gamma * next_potential - potential

        # Normalize the potential-based rewards
        self.reward_normalizer.update(pbim_rewards)
        normalized_pbim_rewards = pbim_rewards / torch.sqrt(
            self.reward_normalizer.var + 1e-8
        )

        # out = torch.zeros_like(normalized_pbim_rewards)
        # out[:, 1:] = normalized_pbim_rewards
        # normalized_pbim_rewards = out

        # Combine with extrinsic rewards
        rewards = buffer_td["rewards"].to(self.device)
        assert rewards.shape == normalized_pbim_rewards.shape, f"Rewards shape: {rewards.shape}, Normalized PBIM rewards shape: {normalized_pbim_rewards.shape}"
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
        buffer_td = buffer_td.view(-1)

        # Reshape for sequence-based processing
        if self.use_shared_encoder_for_icm:
            # Build (t, t+1) pairs along time, flatten over B*(T-1)
            obs_t = buffer_td["observations"][:-1]
            obs_tp = buffer_td["next_observations"][1:]
            act_t = buffer_td["actions"][:-1]

            emb_t = buffer_td["encoder_out"][:-1]
            emb_tp = buffer_td["encoder_out"][1:]
        else:
            obs_t = buffer_td["observations"]
            obs_tp = buffer_td["next_observations"]
            act_t = buffer_td["actions"]
            emb_t = buffer_td["encoder_out"]
            emb_tp = buffer_td["encoder_out"]
            
        num_sequences = obs_t.shape[0]
            
        rewards = buffer_td["rewards"]

        with torch.no_grad():
            potential, next_potential = self.get_potentials(
                action_batch=act_t,
                obs_batch=obs_t,
                next_obs_batch=obs_tp,
                embedded_obs=emb_t,
                embedded_next_obs=emb_tp,
            )

        # Zero out potential for terminal states
        dones = buffer_td["dones"]
        next_potential = next_potential.masked_fill(dones, 0)

        # Compute potential-based shaping reward F(s, s') = gamma * Phi(s') - Phi(s)
        pbim_rewards = self.gamma * next_potential - potential

        # Normalize the potential-based rewards
        # Flatten for normalizer update, then reshape back
        self.reward_normalizer.update(pbim_rewards)
        normalized_pbim_rewards = pbim_rewards / torch.sqrt(
            self.reward_normalizer.var + 1e-8
        )

        # out = torch.zeros_like(rewards)
        # out[:, 1:] = normalized_pbim_rewards
        # normalized_pbim_rewards = out

        # Combine with extrinsic rewards
        combined_rewards = (
            rewards * (1 - self.intrinsic_reward_weight) + self.intrinsic_reward_weight * normalized_pbim_rewards
        )
        combined_rewards = combined_rewards.reshape(
            self.rollout_buffer.capacity, self.num_envs, -1
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

        # Convert actions to appropriate format
        # Use appropriate dtype based on action space type
        dtype = torch.float32 if self.icm.is_continuous_action else torch.long
        action_batch_tensor = self.icm._to_tensor(action_batch_t, dtype=dtype)
        action_input = ICM.actions_to_one_hot(action_batch_tensor, self.action_space)

        (
            _,  # inverse_loss
            _,  # forward_loss
            phi_next_state,
            _,  # icm_hidden_state
            _,  # icm_next_hidden_state
            pred_phi_next_state,  # This is used for our potential function Phi
        ) = self.icm.compute_loss(
            obs_batch=obs_batch_t,
            # action_batch_t=action_batch_t,
            next_obs_batch=next_obs_batch_t,
            action_input=action_input,
            embedded_obs=embedded_obs,
            embedded_next_obs=embedded_next_obs,
            hidden_state_obs=hidden_state_obs,
            hidden_state_next_obs=hidden_state_next_obs,
        )

        # The potential Phi(s,a) is derived from the predicted next state embedding
        potential = pred_phi_next_state.mean(dim=-1, keepdim=True)
        # The next potential Phi(s') is derived from the actual next state embedding
        next_potential = phi_next_state.mean(dim=-1, keepdim=True)

        return potential, next_potential
