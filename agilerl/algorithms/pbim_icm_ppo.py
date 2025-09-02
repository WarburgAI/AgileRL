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
from agilerl.typing import ArrayOrTensor, BPTTSequenceType


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

    This class inherits from the standard ICM module and implements a
    `compute_loss_and_next_state_embedding` method to return the predicted
    next state embedding, which is used as the potential function in PBIM.

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

    def compute_loss_and_next_state_embedding(
        self,
        obs_batch_t: torch.Tensor,
        action_batch_t: torch.Tensor,
        next_obs_batch_t: torch.Tensor,
        embedded_obs: Optional[torch.Tensor] = None,
        embedded_next_obs: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_state_next: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_input: Optional[torch.Tensor] = None,
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

        :param args: Positional arguments passed to the parent's compute_loss_and_next_state_embedding.
        :param kwargs: Keyword arguments passed to the parent's compute_loss_and_next_state_embedding.
        :return: A tuple containing losses, hidden states, and the predicted next state.
        """
        if not action_input:
            action_input = ICM.actions_to_one_hot(action_batch_t, self.action_space)

        (
            phi_state,
            phi_next_state,
            hidden_state,
            hidden_state_next,
        ) = self.embed_obs(
            obs_batch=obs_batch_t,
            next_obs_batch=next_obs_batch_t,
            embedded_obs=embedded_obs,
            embedded_next_obs=embedded_next_obs,
            hidden_state_obs=hidden_state,
            hidden_state_next_obs=hidden_state_next,
        )

        # Get predicted action
        pred_action = self.inverse_model(phi_state, phi_next_state)

        # Get predicted next state
        pred_phi_next_state = self.forward_model(phi_state, action_input)

        # Calculate inverse loss
        if self.is_continuous_action:
            inverse_loss = mse_loss(pred_action, action_input)
        else:
            if isinstance(self.action_space, spaces.Discrete):
                inverse_loss = self.ce_loss_fn(
                    pred_action, torch.argmax(action_input, dim=-1)
                )
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                losses_I = []
                start_idx = 0
                for i, action_size in enumerate(self.inverse_model.action_sizes):
                    end_idx = start_idx + action_size
                    logits_i = pred_action[:, start_idx:end_idx]
                    targets_i = torch.argmax(action_input[:, start_idx:end_idx], dim=-1)
                    loss_i = self.ce_loss_fn(logits_i, targets_i)
                    losses_I.append(loss_i)
                    start_idx = end_idx
                inverse_loss = torch.stack(losses_I).mean()

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
            self.last_potential = torch.zeros(1, device=self.device)
            self.first_potential = False

    def get_potentials(
        self,
        action_batch: Any,
        obs_batch: Optional[Any] = None,
        next_obs_batch: Optional[Any] = None,
        embedded_obs: Optional[torch.Tensor] = None,
        embedded_next_obs: Optional[torch.Tensor] = None,
        hidden_state_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_state_next_obs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the potential values for PBIM.

        This method uses the ICM's forward dynamics model to get the predicted
        and actual next state embeddings, which serve as the potentials.

        :return: A tuple containing the current potential Phi(s,a) and next potential Phi(s').
        """
        dtype = torch.float32 if self.icm.is_continuous_action else torch.long
        action_batch_t = self.icm._to_tensor(action_batch, dtype=dtype)

        obs_batch_t, action_batch_t, next_obs_batch_t = self.to_device(
            obs_batch, action_batch_t, next_obs_batch
        )

        (
            _,  # inverse_loss
            _,  # forward_loss
            phi_next_state,
            _,  # icm_hidden_state
            _,  # icm_next_hidden_state
            pred_phi_next_state,  # This is used for our potential function Phi
        ) = self.icm.compute_loss_and_next_state_embedding(
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
        potential = pred_phi_next_state.mean(dim=-1)
        # The next potential Phi(s') is derived from the actual next state embedding
        next_potential = phi_next_state.mean(dim=-1)

        return potential, next_potential
    
    def get_intrinsic_reward(
        self,
        action_batch: Any,
        dones: Optional[torch.Tensor],
        obs_batch: Optional[Any] = None,
        next_obs_batch: Optional[Any] = None,
        embedded_obs: Optional[torch.Tensor] = None,
        embedded_next_obs: Optional[torch.Tensor] = None,
        hidden_state_obs: Optional[Dict[str, ArrayOrTensor]] = None,
        hidden_state_next_obs: Optional[Dict[str, ArrayOrTensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Function to get the intrinsic reward from the ICM model.

        :param obs: The observation at time t
        :type obs: ArrayOrTensor
        :param action: The action taken at time t
        :type action: ArrayOrTensor
        """
        action_batch = torch.as_tensor(action_batch, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        last_dones = (
            torch.as_tensor(self._last_done, dtype=torch.bool, device=self.device)
            if self._last_done is not None
            else torch.ones_like(dones, dtype=torch.bool, device=self.device)
        )
        obs_batch = (
            torch.as_tensor(obs_batch, device=self.device)
            if obs_batch is not None
            else None
        )
        next_obs_batch = (
            torch.as_tensor(next_obs_batch, device=self.device)
            if next_obs_batch is not None
            else None
        )
        embedded_obs = (
            torch.as_tensor(embedded_obs, device=self.device)
            if embedded_obs is not None
            else None
        )
        embedded_next_obs = (
            torch.as_tensor(embedded_next_obs, device=self.device)
            if embedded_next_obs is not None
            else None
        )
        hidden_state_obs = (
            {k : torch.as_tensor(v, device=self.device) for k, v in hidden_state_obs.items()}
            if hidden_state_obs is not None
            else None
        )
        hidden_state_next_obs = (
            {k : torch.as_tensor(v, device=self.device) for k, v in hidden_state_next_obs.items()}
            if hidden_state_next_obs is not None
            else None
        )
        
        with torch.no_grad():
            instrinsic_reward, hidden_state, next_hidden_state = super().get_intrinsic_reward(
                action_batch=action_batch,
                obs_batch=obs_batch,
                next_obs_batch=next_obs_batch,
                embedded_obs=embedded_obs,
                embedded_next_obs=embedded_next_obs,
                hidden_state_obs=hidden_state_obs,
                hidden_state_next_obs=hidden_state_next_obs,
            )
            
            if not self.first_potential:
                embedded_start_obs, hidden_start_state = self.icm.encoder(
                    torch.zeros_like(obs_batch, device=self.device, dtype=obs_batch.dtype),
                    {k: torch.zeros_like(v, device=self.device, dtype=v.dtype) for k, v in hidden_state.items()}
                    if self.icm.is_recurrent else None
                )

                last_potential, _, _ = super().get_intrinsic_reward(
                    action_batch=action_batch,
                    obs_batch=torch.zeros_like(obs_batch, device=self.device, dtype=obs_batch.dtype),
                    next_obs_batch=obs_batch,
                    embedded_obs=embedded_start_obs,
                    embedded_next_obs=embedded_obs,
                    hidden_state_obs=hidden_start_state,
                    hidden_state_next_obs=hidden_state,
                )
                torch.div(last_potential, self.icm.intrinsic_reward_weight, out=last_potential) # rescale for stats calculation
                last_potential = last_potential.masked_fill(last_dones, 0) # mask out the last potential for the first step explicitly for clarity (of course it doesn't matter for the first step but it's good to be explicit)
                last_potential = last_potential.mul_(self.gamma) - 0 # explicitly set last_potential to 0 for the first step for clarity
                self.reward_normalizer.update(last_potential)
                denom = torch.sqrt(self.reward_normalizer.var + 1e-8)
                torch.div(last_potential, denom, out=last_potential)
                last_potential = last_potential.mul_(self.icm.intrinsic_reward_weight) # rescale for stats calculation
                
                self.first_potential = True
            else:
                last_potential = self.last_potential
                
            # Zero out potential for terminal states, as per PBRS for episodic tasks
            next_potential = instrinsic_reward.masked_fill(dones, 0)

            # Compute potential-based shaping reward F(s, s') = gamma * Phi(s') - Phi(s)
            next_potential.mul_(self.gamma)
            torch.add(next_potential, last_potential, alpha=-1.0, out=next_potential)  # next_potential - potential
            # next_potential should now be called pbim_rewards, but we're doing it in-place to avoid temporaries
    
            # Normalize the potential-based rewards (avoid temporaries)
            self.reward_normalizer.update(next_potential)
            denom = torch.sqrt(self.reward_normalizer.var + 1e-8)
            torch.div(next_potential, denom, out=next_potential)
            # next_potential is now normalized pbim_rewards
            
            self.last_potential = next_potential
            return self.icm.intrinsic_reward_weight * next_potential, None, None
            
            
        # return self.icm.get_intrinsic_reward(
        #     action_batch,
        #     obs_batch,
        #     next_obs_batch,
        #     embedded_obs,
        #     embedded_next_obs,
        #     hidden_state_obs,
        #     hidden_state_next_obs,
        # )
