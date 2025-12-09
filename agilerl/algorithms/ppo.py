import copy
import gc
import math
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from gymnasium import spaces
from tensordict import TensorDict
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks import EvolvableNetwork, StochasticActor
from agilerl.networks.value_networks import ValueNetwork
from agilerl.typing import ArrayOrTensor, BPTTSequenceType, ExperiencesType, GymEnvType
from agilerl.utils.algo_utils import (
    make_safe_deepcopies,
    obs_channels_to_first,
    share_encoder_parameters,
)
from agilerl.utils.metrics import MetricsTracker, TimingTracker
from agilerl.wrappers.utils import RunningMeanStd


class PPO(RLAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm.

    Paper: https://arxiv.org/abs/1707.06347v2

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Network configuration, defaults to None
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param gae_lambda: Lambda for general advantage estimation, defaults to 0.95
    :type gae_lambda: float, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param action_std_init: Initial action standard deviation, defaults to 0.0
    :type action_std_init: float, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param ent_coef: Entropy coefficient, defaults to 0.01
    :type ent_coef: float, optional
    :param vf_coef: Value function coefficient, defaults to 0.5
    :type vf_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param target_kl: Target KL divergence threshold, defaults to None
    :type target_kl: float, optional
    :param normalize_images: Flag to normalize images, defaults to True
    :type normalize_images: bool, optional
    :param normalize_rewards: Flag to normalize scalar rewards using a running mean/std, defaults to True
    :type normalize_rewards: bool, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param share_encoders: Flag to share encoder parameters between actor and critic, defaults to False
    :type share_encoders: bool, optional
    :param num_envs: Number of parallel environments, defaults to 1
    :type num_envs: int, optional
    :param use_rollout_buffer: Flag to use the rollout buffer instead of tuple experiences, defaults to False
    :type use_rollout_buffer: bool, optional
    :param recurrent: Flag to use hidden states for recurrent policies, defaults to False
    :type recurrent: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param bptt_sequence_type: Type of sequence for BPTT learning, defaults to BPTTSequenceType.CHUNKED
    :type bptt_sequence_type: BPTTSequenceType, optional
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
        vf_clip_param: Optional[float] = None,
        optimizer: str = "adam",
        optimizer_eps: float = 1e-5,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        normalize_images: bool = True,
        normalize_rewards: bool = False,
        update_epochs: int = 4,
        actor_network: Optional[EvolvableModule] = None,
        critic_network: Optional[EvolvableModule] = None,
        share_encoders: bool = True,
        num_envs: int = 1,
        use_rollout_buffer: bool = False,
        rollout_buffer_config: Optional[Dict[str, Any]] = {},
        recurrent: bool = False,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True,
        bptt_sequence_type: BPTTSequenceType = BPTTSequenceType.CHUNKED,
        torch_compiler: Optional[Any] = None,
        use_experimental_distributions: Optional[bool] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            normalize_images=normalize_images,
            torch_compiler=torch_compiler,
            name="PPO",
        )

        self.normalize_rewards = normalize_rewards
        self.reward_rms = RunningMeanStd(epsilon=1e-4, device="cpu")

        assert learn_step >= 1, "Learn step must be greater than or equal to one."
        assert isinstance(learn_step, int), "Learn step must be an integer."
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
        assert isinstance(gae_lambda, (float, int)), "Lambda must be a float."
        assert gae_lambda >= 0, "Lambda must be greater than or equal to zero."
        assert isinstance(
            action_std_init, (float, int)
        ), "Action standard deviation must be a float."
        assert (
            action_std_init >= 0
        ), "Action standard deviation must be greater than or equal to zero."
        assert isinstance(
            clip_coef, (float, int)
        ), "Clipping coefficient must be a float."
        assert (
            clip_coef >= 0
        ), "Clipping coefficient must be greater than or equal to zero."
        assert isinstance(
            ent_coef, (float, int)
        ), "Entropy coefficient must be a float."
        assert (
            ent_coef >= 0
        ), "Entropy coefficient must be greater than or equal to zero."
        assert isinstance(
            vf_coef, (float, int)
        ), "Value function coefficient must be a float."
        assert (
            vf_coef >= 0
        ), "Value function coefficient must be greater than or equal to zero."
        assert isinstance(
            max_grad_norm, (float, int)
        ), "Maximum norm for gradient clipping must be a float."
        assert (
            max_grad_norm >= 0
        ), "Maximum norm for gradient clipping must be greater than or equal to zero."
        assert (
            isinstance(target_kl, (float, int)) or target_kl is None
        ), "Target KL divergence threshold must be a float."
        if target_kl is not None:
            assert (
                target_kl >= 0
            ), "Target KL divergence threshold must be greater than or equal to zero."
        assert isinstance(
            update_epochs, int
        ), "Policy update epochs must be an integer."
        assert (
            update_epochs >= 1
        ), "Policy update epochs must be greater than or equal to one."
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        # New parameters for using RolloutBuffer
        assert isinstance(
            use_rollout_buffer, bool
        ), "Use rollout buffer flag must be boolean value True or False."
        assert isinstance(
            recurrent, bool
        ), "Has hidden states flag must be boolean value True or False."
        assert isinstance(
            bptt_sequence_type, BPTTSequenceType
        ), "bptt_sequence_type must be a BPTTSequenceType enum value."

        if not use_rollout_buffer:
            warnings.warn(
                (
                    "DeprecationWarning: 'use_rollout_buffer=False' is deprecated and will be removed in a future release. "
                    "The PPO implementation now expects 'use_rollout_buffer=True' for improved performance, "
                    "cleaner support for recurrent policies, and easier integration with custom environments. "
                    "Please update your code to use 'use_rollout_buffer=True' and, if you require recurrent policies, set 'recurrent=True'.\n"
                    "Refer to the documentation for migration instructions and further details."
                ),
                DeprecationWarning,
                stacklevel=2,
            )

        self.recurrent = recurrent
        self.use_rollout_buffer = use_rollout_buffer
        self.net_config = net_config

        if self.recurrent:
            if not self.use_rollout_buffer:
                raise ValueError("use_rollout_buffer must be True if recurrent=True.")
            net_config_dict = self.net_config if self.net_config is not None else {}
            self.max_seq_len = net_config_dict.get("encoder_config", {}).get(
                "max_seq_len", None
            )
            if self.max_seq_len is None:
                raise ValueError(
                    "max_seq_len must be provided in net_config['encoder_config'] if recurrent=True."
                )
        else:
            self.max_seq_len = None

        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.learn_step = learn_step
        self.mut = mut
        self.gae_lambda = gae_lambda
        self.action_std_init = action_std_init
        self.clip_coef = clip_coef
        self.vf_clip_param = clip_coef if vf_clip_param is None else vf_clip_param
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.num_envs = num_envs
        self.rollout_buffer_config = rollout_buffer_config
        self.bptt_sequence_type = bptt_sequence_type
        self.use_experimental_distributions = (
            use_experimental_distributions
            if use_experimental_distributions is not None
            else os.environ.get("USE_EXPERIMENTAL_DISTRIBUTIONS", "false").lower()
            == "true"
        )

        if actor_network is not None and critic_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"Passed actor network is of type {type(actor_network)}, but must be of type EvolvableModule."
                )
            if not isinstance(critic_network, EvolvableModule):
                raise TypeError(
                    f"Passed critic network is of type {type(critic_network)}, but must be of type EvolvableModule."
                )

            self.actor, self.critic = make_safe_deepcopies(
                actor_network, critic_network
            )
        else:
            net_config_dict = {} if self.net_config is None else self.net_config

            critic_net_config = copy.deepcopy(net_config_dict)

            head_config = net_config_dict.get("head_config", None)
            if head_config is not None:
                critic_head_config = copy.deepcopy(head_config)
                critic_head_config["output_activation"] = None
                critic_net_config.pop("squash_output", None)
            else:
                critic_head_config = MlpNetConfig(hidden_size=[16])

            critic_net_config["head_config"] = critic_head_config

            self.actor = StochasticActor(
                self.observation_space,
                self.action_space,
                action_std_init=self.action_std_init,
                device=self.device,
                recurrent=self.recurrent,
                encoder_name=("shared_encoder" if share_encoders else "actor_encoder"),
                use_experimental_distribution=self.use_experimental_distributions,
                **net_config_dict,
            )

            self.critic = ValueNetwork(
                self.observation_space,
                device=self.device,
                recurrent=self.recurrent,
                encoder_name=("shared_encoder" if share_encoders else "critic_encoder"),
                **critic_net_config,
            )

        # Share encoders between actor and critic
        self.share_encoders = share_encoders
        if self.share_encoders and all(
            isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]
        ):
            self.share_encoder_parameters()
            # Need to register a mutation hook that does this after every mutation
            self.register_mutation_hook(self.share_encoder_parameters)

        optim_cls = (
            optim.Adam
            if optimizer == "adam"
            else (
                optim.AdamW
                if optimizer == "adamw"
                else optim.Muon if optimizer == "muon" else None
            )
        )
        if optim_cls is None:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        self.optimizer = OptimizerWrapper(
            optim_cls,
            networks=[self.actor, self.critic],
            lr=self.lr,
            optimizer_kwargs={"eps": optimizer_eps},
        )

        # Initialize rollout buffer if enabled
        if self.use_rollout_buffer:
            self.create_rollout_buffer()
            # Need to register a mutation hook that does this after every mutation (e.g. the batch size, sequence length, etc. have changed)
            self.register_mutation_hook(self.create_rollout_buffer)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        self.register_network_group(NetworkGroup(eval_network=self.critic))

        self.hidden_state = None
        self._last_obs = None
        self._last_done = None
        self._last_scores = None
        self._last_info = None

        # Initialize metrics trackers
        self.learn_metrics = MetricsTracker()
        self.timing_tracker = TimingTracker()
        self.total_learn_time = 0.0
        self.total_collection_time = 0.0
        self.last_learn_time = 0.0
        self.last_collection_time = 0.0

    def normalize_reward(self, reward: ArrayOrTensor) -> np.ndarray:
        """Normalize extrinsic rewards using a running mean and variance."""
        if not self.normalize_rewards:
            return np.asarray(reward, dtype=np.float32)

        reward_np = np.asarray(reward, dtype=np.float32)
        if reward_np.size == 0:
            return reward_np

        flat = reward_np.reshape(-1)
        reward_tensor = torch.from_numpy(flat).to(self.reward_rms.mean.device)
        self.reward_rms.update(reward_tensor)
        std = math.sqrt(self.reward_rms.var.item() + 1e-8)
        mean = self.reward_rms.mean.item()

        if std <= 0.0:
            normalized_flat = flat - mean
        else:
            normalized_flat = (flat - mean) / std

        return normalized_flat.reshape(reward_np.shape).astype(np.float32)

    def share_encoder_parameters(self) -> None:
        """Shares the encoder parameters between the actor and critic."""
        if all(isinstance(net, EvolvableNetwork) for net in [self.actor, self.critic]):
            share_encoder_parameters(self.actor, self.critic)
        else:
            warnings.warn(
                "Encoder sharing is disabled as actor or critic is not an EvolvableNetwork."
            )

    def create_rollout_buffer(self) -> None:
        """Creates a rollout buffer with the current configuration."""
        self.rollout_buffer = RolloutBuffer(
            capacity=self.learn_step,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            num_envs=self.num_envs,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            recurrent=self.recurrent,
            # recurrent specific parameters
            hidden_state_architecture=(
                self.get_hidden_state_architecture() if self.recurrent else None
            ),
            max_seq_len=self.max_seq_len if self.recurrent else None,
            **self.rollout_buffer_config,
        )

    def _get_action_and_values(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[
            Dict[str, ArrayOrTensor]
        ] = None,  # Hidden state is a dict for recurrent policies
        *,
        sample: bool = True,
        deterministic: bool = False,
        compute_values: bool = False,
    ) -> Tuple[
        ArrayOrTensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Dict[str, ArrayOrTensor]],
    ]:
        """
        Returns the next action to take in the environment and the values.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: Optional[ArrayOrTensor]
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: Optional[Dict[str, ArrayOrTensor]]
        :param sample: Whether to sample an action, defaults to True
        :type sample: bool
        :param deterministic: Whether to return a deterministic action, defaults to False
        :type deterministic: bool, optional
        :param compute_values: Whether to compute values from critic, defaults to False for speed
        :type compute_values: bool
        :return: Action, log probability, entropy, state values (or None), and (if recurrent) next hidden state
        :rtype: Tuple[ArrayOrTensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, ArrayOrTensor]]]
        """
        if hidden_state is not None:
            latent_pi, next_hidden_actor = self.actor.extract_features(
                obs, hidden_state=hidden_state
            )
            action, log_prob, entropy = self.actor.forward_head(
                latent_pi,
                action_mask=action_mask,
                sample=sample,
                deterministic=deterministic,
            )

            # Start with actor's next hidden state
            next_hidden_combined: Dict[str, torch.Tensor] = next_hidden_actor

            if compute_values:
                if self.share_encoders:
                    values = self.critic.forward_head(latent_pi).squeeze(-1)
                else:
                    # If not sharing, critic might have its own hidden state components or update existing ones
                    values, next_hidden_critic = self.critic(
                        obs, hidden_state=hidden_state
                    )  # Pass original hidden_state
                    values = values.squeeze(-1)

                    # Merge if critic returns its own next_hidden
                    if next_hidden_critic is not None:
                        next_hidden_combined.update(next_hidden_critic)
            else:
                values = None

            return action, log_prob, entropy, values, next_hidden_combined
        else:
            latent_pi = self.actor.extract_features(obs)
            action, log_prob, entropy = self.actor.forward_head(
                latent_pi,
                action_mask=action_mask,
                sample=sample,
                deterministic=deterministic,
            )

            if compute_values:
                if self.share_encoders:
                    values = self.critic.forward_head(latent_pi).squeeze(-1)
                else:
                    critic_output = self.critic(obs)
                    # Handle case where critic returns tuple (recurrent critic with hidden_state=None)
                    if isinstance(critic_output, tuple):
                        values = critic_output[0].squeeze(-1)
                    else:
                        values = critic_output.squeeze(-1)
            else:
                values = None

            return action, log_prob, entropy, values, None

    def get_hidden_state_architecture(self) -> Dict[str, Tuple[int, ...]]:
        """Get the hidden state architecture for the environment.

        :return: Dictionary describing the hidden state architecture (name to shape)
        :rtype: Dict[str, Tuple[int, ...]]
        """
        return {
            k: v.shape for k, v in self.get_initial_hidden_state(self.num_envs).items()
        }

    def get_initial_hidden_state(self, num_envs: int = 1) -> Dict[str, ArrayOrTensor]:
        """Get the initial hidden state for the environment.

        The hidden states are generally cached on a per Module basis.
        The reason the Cache is per Module is because the user might want to have a custom initialization for the hidden states.

        :param num_envs: Number of environments, defaults to 1
        :type num_envs: int, optional
        :return: Initial hidden state dictionary
        :rtype: Dict[str, ArrayOrTensor]
        """
        # Return a batch of initial hidden states
        # Flat map them into "actor_*" and "critic_*" (if not sharing encoders)
        hidden = TensorDict()

        actor_hidden = self.actor.initialize_hidden_state(
            device=self.device, batch_size=num_envs
        )
        hidden.update(actor_hidden)

        # also add the critic hidden state if not sharing encoders
        if not self.share_encoders:
            critic_hidden = self.critic.initialize_hidden_state(
                device=self.device, batch_size=num_envs
            )
            hidden.update(critic_hidden)

        return hidden

    def evaluate_actions(
        self,
        obs: ArrayOrTensor,
        actions: ArrayOrTensor,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the actions.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param actions: Actions to evaluate
        :type actions: ArrayOrTensor
        :param hidden_state: Hidden state for recurrent policies, defaults to None. Expected shape: dict with tensors of shape (batch_size, 1, hidden_size).
        :type hidden_state: Optional[Dict[str, ArrayOrTensor]]
        :return: Log probability, entropy, and state values
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs = self.preprocess_observation(obs)

        # Get values from actor-critic
        _, _, entropy, values, _ = self._get_action_and_values(
            obs, hidden_state=hidden_state, sample=False
        )

        log_prob = self.actor.action_log_prob(actions)

        # Compute proper entropy if not provided
        if entropy is None:
            # Try to compute entropy from the actor's head
            try:
                latent = self.actor.extract_features(obs, hidden_state=hidden_state)
                entropy = self.actor.head_net.entropy_from_latent(
                    latent, action_mask=None
                )
            except (AttributeError, NotImplementedError):
                # Fallback: this is not accurate entropy but a rough approximation
                entropy = -log_prob.mean()

        return log_prob, entropy, values

    def get_action(
        self,
        obs: ArrayOrTensor,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
        deterministic: bool = False,
        compute_values: bool = False,
    ) -> Union[
        Tuple[
            np.ndarray,  # action
            np.ndarray,  # log_prob
            np.ndarray,  # entropy
            Optional[np.ndarray],  # values (None if compute_values=False)
            Optional[Dict[str, ArrayOrTensor]],  # next_hidden_state
        ],
        Tuple[
            np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]
        ],  # non-recurrent case
    ]:
        """Returns the next action to take in the environment.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: ArrayOrTensor
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: Optional[ArrayOrTensor]
        :param hidden_state: Hidden state for recurrent policies, defaults to None
        :type hidden_state: Optional[Dict[str, ArrayOrTensor]]
        :param deterministic: Boolean specifying whether to desired action is stochastic or deterministic, defaults to False
        :type deterministic: bool, optional
        :param compute_values: Whether to compute values from critic, defaults to False for speed
        :type compute_values: bool
        :return: Action, log probability, entropy, state values (or None), and (if recurrent) next hidden state
        :rtype: Union[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, ArrayOrTensor]]], Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]
        """
        obs = self.preprocess_observation(obs)
        with torch.no_grad():
            action, log_prob, entropy, values, next_hidden = (
                self._get_action_and_values(
                    obs,
                    action_mask,
                    hidden_state,
                    sample=not deterministic,
                    deterministic=deterministic,
                    compute_values=compute_values,
                )
            )

        # Compute proper entropy if not provided
        if entropy is None:
            # Try to compute entropy from the actor's head
            try:
                with torch.no_grad():
                    # obs is already preprocessed above, reuse it directly
                    latent = self.actor.extract_features(
                        obs, hidden_state=hidden_state
                    )
                    entropy = self.actor.head_net.entropy_from_latent(
                        latent, action_mask=action_mask
                    )
            except (AttributeError, NotImplementedError):
                # Fallback: this is not accurate entropy but a rough approximation
                entropy = -log_prob.mean()

        # Clip to action space during inference
        action_np = action.cpu().data.numpy()
        if not self.training and isinstance(self.action_space, spaces.Box):
            if self.actor.squash_output:
                action_np = self.actor.scale_action(action_np)
            else:
                action_np = np.clip(
                    action_np, self.action_space.low, self.action_space.high
                )

        log_prob_np = (
            log_prob.cpu().data.numpy()
            if log_prob is not None
            else np.zeros(action_np.shape[0], dtype=np.float32)
        )
        entropy_np = (
            entropy.cpu().data.numpy()
            if entropy is not None
            else np.zeros(action_np.shape[0], dtype=np.float32)
        )
        values_np = values.cpu().data.numpy() if values is not None else None

        if self.recurrent:
            return (
                action_np,
                log_prob_np,
                entropy_np,
                values_np,
                next_hidden if next_hidden is not None else None,
            )
        else:
            return (
                action_np,
                log_prob_np,
                entropy_np,
                values_np,
            )

    @staticmethod
    def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate explained variance.

        :param y_pred: Predicted values (shape: (N,))
        :param y_true: True returns (shape: (N,))
        :return: Explained variance ratio
        """
        var_y = torch.var(y_true)
        # Avoid CPU sync; return a scalar tensor on the same device
        ev = 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        # When variance is ~0, clamp to 0 to avoid spurious values
        return torch.clamp(ev, min=0.0, max=1.0)

    def compute_loss(
        self,
        obs,
        actions,
        old_log_probs,
        advantages,
        returns,
        hidden_state=None,
        old_values=None,
        learn_by_bptt=False,
        seq_len=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute the loss for the actor and critic networks."""
        action_mask = kwargs.get("action_mask", None)  # optional
        if learn_by_bptt:
            if seq_len is None:
                seq_len = self.max_seq_len
            # Preserve [B, T, ...] shapes for recurrent sequence processing.
            # Only ensure tensors are on the correct device without reshaping.
            if isinstance(obs, dict):
                obs = {
                    k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).to(
                        self.device
                    )
                    for k, v in obs.items()
                }
            elif isinstance(obs, torch.Tensor):
                obs = obs.to(self.device)
            else:  # numpy or other array-like
                obs = torch.as_tensor(obs, device=self.device)
            # ---- Features / values for the whole sequence (do NOT resample actions) ----
            # (we ignore returned sampled actions/log_probs)
            _, _, entropies, features_seq, _ = self.actor.sequence_forward(
                obs, hidden_state, action_mask=action_mask
            )
            # Values from critic
            if self.share_encoders:
                # Ensure critic head receives 2D [B*T, latent] input, then reshape back to [B, T]
                B, T = features_seq.shape[:2]
                flat_feat_for_critic = features_seq.reshape(B * T, -1)
                new_values = (
                    self.critic.forward_head(flat_feat_for_critic)
                    .squeeze(-1)
                    .view(B, T)
                )
            else:
                # Observations already preprocessed above for consistency
                new_values, _ = self.critic.sequence_forward(obs, hidden_state)
                new_values = new_values.squeeze(-1)  # [B,T]
            B, T = features_seq.shape[:2]
            flat_feat = features_seq.reshape(B * T, -1)
            # Flatten actions (and mask if provided) to match latent
            flat_act = actions.reshape(B * T, -1)
            # Fix discrete action shape - squeeze extra dimension
            if isinstance(self.action_space, spaces.Discrete):
                flat_act = flat_act.view(-1)  # (B*T,)
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                flat_act = flat_act.long()  # Ensure long dtype for MultiDiscrete
            flat_mask = (
                action_mask.reshape(B * T, -1) if action_mask is not None else None
            )
            # ---- Log-prob of BUFFER actions under CURRENT policy ----
            new_log_probs = self.actor.head_net.log_prob_from_latent(
                flat_feat, flat_act, action_mask=flat_mask
            ).view(B, T)
            # ---- PPO clipped objective ----
            log_ratio = new_log_probs - old_log_probs  # [B,T]
            ratio = torch.exp(log_ratio)
            policy_loss1 = -advantages * ratio
            policy_loss2 = -advantages * torch.clamp(
                ratio, 1 - self.clip_coef, 1 + self.clip_coef
            )
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()
            # Value loss (clip against old_values if provided; shapes [B,T])
            if old_values is not None:
                v_loss_unclipped = (new_values - returns) ** 2
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.vf_clip_param, self.vf_clip_param
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()
            # Entropy: use analytic entropies if available, else fallback to -log_prob mean
            if entropies is None:
                entropy_loss = -new_log_probs.mean()
            else:
                entropy_loss = -entropies.mean()
            loss = (
                policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            )
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > self.clip_coef).float().mean()
            return {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            }
        # ---------------------- FLAT (non-BPTT) path ----------------------
        # Compute latent features once (no resampling)
        latent = self.actor.extract_features(obs, hidden_state=hidden_state)
        # Log-prob / entropy under CURRENT policy with mask-aware head
        new_log_prob_t = self.actor.head_net.log_prob_from_latent(
            latent, actions, action_mask=action_mask
        )
        entropy_t = self.actor.head_net.entropy_from_latent(
            latent, action_mask=action_mask
        )
        # Values (reuse latent if encoders shared)
        if self.share_encoders:
            new_value_t = self.critic.forward_head(latent).squeeze(-1)
        else:
            new_value_t = self.critic(obs).squeeze(-1)

        ratio = torch.exp(new_log_prob_t - old_log_probs)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        if old_values is not None:
            v_loss_unclipped = (new_value_t - returns) ** 2
            v_clipped = old_values + torch.clamp(
                new_value_t - old_values, -self.vf_clip_param, self.vf_clip_param
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((new_value_t - returns) ** 2).mean()
        entropy_loss = -entropy_t.mean()
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        with torch.no_grad():
            log_ratio = new_log_prob_t - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_coef).float().mean()
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def learn(self, experiences: Optional[ExperiencesType] = None) -> Dict[str, float]:
        """Updates agent network parameters to learn from experiences.

        :param experiences: Tuple of batched states, actions, log_probs, rewards, dones, values, next_state, next_done.
                            If use_rollout_buffer=True and experiences=None, uses data from rollout buffer.
        :type experiences: Optional[ExperiencesType]
        :return: Dictionary of metrics including total_loss, policy_loss, value_loss, entropy_loss, approx_kl, and clip_fraction.
        :rtype: Dict[str, float]
        """
        # Reset metrics and start timing
        self.learn_metrics.reset()
        self.timing_tracker.start_timer("learn_total")

        if self.use_rollout_buffer:
            # NOTE: we are still allowing experiences to be passed in for backwards compatibility
            # but we will remove this in a future releases.
            # i.e. it's possible to do one learn with rollouts, then another with experiences on the same agent
            if experiences is None:
                # Learn from the internal rollout buffer
                if (
                    self.recurrent
                    and self.max_seq_len is not None
                    and self.max_seq_len > 0
                ):
                    self._learn_from_rollout_buffer_bptt()
                else:
                    self._learn_from_rollout_buffer_flat()
        else:
            self._deprecated_learn_from_experiences(experiences)

        # Finalize timing and collect metrics
        self.last_learn_time = self.timing_tracker.end_timer("learn_total")
        self.total_learn_time += self.last_learn_time

        # clear cache
        # gc.collect()

        # Clear CUDA cache after training to free memory
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # Combine all metrics
        metrics = self.learn_metrics.get_all_averages("learn/")
        metrics.update(
            {
                "time/last_learn_time": self.last_learn_time,
                "time/total_learn_time": self.total_learn_time,
                "time/last_collection_time": self.last_collection_time,
                "time/total_collection_time": self.total_collection_time,
            }
        )
        metrics.update(self.timing_tracker.get_metrics())

        return metrics

    def _deprecated_learn_from_experiences(
        self, experiences: ExperiencesType
    ) -> Dict[str, float]:
        """Deprecated method for learning from experiences tuple format.

        This method is deprecated and will be removed in a future release. The PPO implementation
        now uses a rollout buffer for improved performance, cleaner support for recurrent policies,
        and easier integration with custom environments.

        To migrate:
        1. Set use_rollout_buffer=True when creating PPO agent
        2. If using recurrent policies, set recurrent=True
        3. Use collect_rollouts() to gather experiences instead of passing experiences tuple
        4. Call learn() without arguments to train on collected rollouts
        """
        raise NotImplementedError(
            "This method is now out of date. Use learn() instead."
        )

    def _learn_from_rollout_buffer_flat(
        self, buffer_td_external: Optional[TensorDict] = None
    ) -> None:
        """Learning procedure using flattened samples (no BPTT)."""
        if buffer_td_external is not None:
            buffer_td = buffer_td_external
        else:
            # .get_tensor_batch() returns a TensorDict on the specified device
            with self.timing_tracker.time_context("get_tensor_batch_time"):
                buffer_td = self.rollout_buffer.get_tensor_batch(
                    device=self.device,
                    include_keys=[
                        "observations",
                        "actions",
                        "log_probs",
                        "advantages",
                        "returns",
                        "values",
                        "action_masks",
                        "hidden_states",
                    ],
                )

        if buffer_td.is_empty():
            warnings.warn("Buffer data is empty. Skipping learning step.")
            return

        observations = buffer_td["observations"]
        advantages = buffer_td["advantages"]

        # Normalize advantages
        with self.timing_tracker.time_context("advantage_normalization_time"):
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages.sub_(adv_mean).div_(adv_std + 1e-8)

        batch_size = self.batch_size
        num_samples = observations.size(0)  # Total number of samples in the buffer
        # Release the large observations view reference; we'll slice from buffer_td directly in minibatches
        del observations
        indices = np.arange(num_samples)

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            num_minibatches_this_epoch = 0

            # Accumulate metrics as tensors to avoid per-minibatch CPU sync
            sum_total_loss = torch.zeros((), device=self.device)
            sum_policy_loss = torch.zeros((), device=self.device)
            sum_value_loss = torch.zeros((), device=self.device)
            sum_entropy_loss = torch.zeros((), device=self.device)
            sum_approx_kl = torch.zeros((), device=self.device)
            sum_clip_fraction = torch.zeros((), device=self.device)
            sum_ev = torch.zeros((), device=self.device)
            sum_actor_norm = torch.zeros((), device=self.device)
            sum_critic_norm = torch.zeros((), device=self.device)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                minibatch_indices = indices[start_idx:end_idx]

                # Slice the TensorDict to get the minibatch
                minibatch_td = buffer_td[minibatch_indices]

                mb_obs = minibatch_td["observations"]
                mb_actions = minibatch_td["actions"]
                mb_old_log_probs = minibatch_td["log_probs"]
                mb_advantages = advantages[
                    minibatch_indices
                ]  # Use globally normalized advantages
                mb_returns = minibatch_td["returns"]
                mb_old_values = minibatch_td["values"]
                mb_action_masks = (
                    minibatch_td.get("action_masks")
                    if "action_masks" in minibatch_td.keys(include_nested=True)
                    else None
                )

                # Prepare hidden state if recurrent, then drop the container ASAP
                eval_hidden_state = None
                if self.recurrent:
                    if "hidden_states" in minibatch_td.keys(include_nested=True):
                        mb_hidden_states_td = minibatch_td.get("hidden_states")
                        eval_hidden_state = {
                            k: v.permute(1, 0, 2).contiguous().detach()
                            for k, v in mb_hidden_states_td.items()
                        }
                        # Free the raw hidden states TD after conversion
                        del mb_hidden_states_td
                    else:
                        warnings.warn(
                            "Recurrent policy, but no hidden_states found in minibatch_td for flat learning."
                        )

                # Free the minibatch container before compute to lower peak memory
                del minibatch_td

                with self.timing_tracker.time_context("loss_calculation_time"):
                    loss_dict = self.compute_loss(
                        mb_obs,
                        mb_actions,
                        mb_old_log_probs,
                        mb_advantages,
                        mb_returns,
                        hidden_state=eval_hidden_state,
                        old_values=mb_old_values,
                        learn_by_bptt=False,
                        action_mask=mb_action_masks,
                    )
                loss = loss_dict["loss"]

                # Accumulate metrics (tensor scalars, detached)
                sum_policy_loss += loss_dict["policy_loss"].detach()
                sum_value_loss += loss_dict["value_loss"].detach()
                sum_entropy_loss += loss_dict["entropy_loss"].detach()
                sum_approx_kl += loss_dict["approx_kl"].detach()
                sum_clip_fraction += loss_dict["clip_fraction"].detach()

                # Compute EV using old values vs returns (no extra forward)
                with torch.no_grad():
                    ev = self._explained_variance(
                        mb_old_values.reshape(-1), mb_returns.reshape(-1)
                    )
                    sum_ev += ev

                # Release non-required tensors before backward
                if mb_action_masks is not None:
                    del mb_action_masks

                with self.timing_tracker.time_context("backward_pass_time"):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_norm = clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    critic_norm = clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                # Track only minibatch count; aggregation happens after loop
                sum_total_loss += loss.detach()
                sum_actor_norm += actor_norm.detach()
                sum_critic_norm += critic_norm.detach()

                num_minibatches_this_epoch += 1

                # Clean up minibatch tensors to free memory
                del (
                    mb_obs,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    mb_old_values,
                )
                if eval_hidden_state is not None:
                    del eval_hidden_state
                del loss_dict, loss

                # Check KL divergence for early stopping - only check every 4 minibatches to reduce GPU sync overhead
                should_stop = False
                if self.target_kl is not None and num_minibatches_this_epoch % 4 == 0:
                    # Use accumulated KL average instead of single minibatch
                    avg_approx_kl = (sum_approx_kl / max(1, num_minibatches_this_epoch)).item()
                    should_stop = avg_approx_kl > self.target_kl
                    if should_stop:
                        warnings.warn(
                            f"Flat learning: KL divergence {avg_approx_kl:.4f} exceeded target {self.target_kl}. Stopping update for this epoch."
                        )
                        break  # Break from minibatch loop for this epoch

            # After all minibatches for this epoch, log averaged metrics once
            # Batch all GPU->CPU transfers into one call to reduce sync overhead
            denom = max(1, num_minibatches_this_epoch)
            inv = 1.0 / denom
            # Stack all metrics into a single tensor for one CPU transfer
            metrics_tensor = torch.stack([
                sum_total_loss * inv,
                sum_policy_loss * inv,
                sum_value_loss * inv,
                sum_entropy_loss * inv,
                sum_approx_kl * inv,
                sum_clip_fraction * inv,
                sum_ev * inv,
                sum_actor_norm * inv,
                sum_critic_norm * inv,
            ]).detach().cpu()

            self.learn_metrics.add("total_loss", float(metrics_tensor[0]))
            self.learn_metrics.add("policy_loss", float(metrics_tensor[1]))
            self.learn_metrics.add("value_loss", float(metrics_tensor[2]))
            self.learn_metrics.add("entropy_loss", float(metrics_tensor[3]))
            self.learn_metrics.add("approx_kl", float(metrics_tensor[4]))
            self.learn_metrics.add("clip_fraction", float(metrics_tensor[5]))
            self.learn_metrics.add("explained_variance", float(metrics_tensor[6]))
            self.learn_metrics.add("actor_grad_norm", float(metrics_tensor[7]))
            self.learn_metrics.add("critic_grad_norm", float(metrics_tensor[8]))

        # Free large references after training step
        del buffer_td
        del advantages

    def _learn_from_rollout_buffer_bptt(self) -> None:
        """Learning procedure using truncated BPTT for recurrent networks."""
        seq_len = self.max_seq_len

        buffer_actual_size = (
            self.rollout_buffer.capacity
            if self.rollout_buffer.full
            else self.rollout_buffer.pos
        )
        if buffer_actual_size < seq_len:
            warnings.warn(
                f"Buffer size {buffer_actual_size} is less than seq_len {seq_len}. Skipping BPTT learning step."
            )
            return

        # Normalize advantages globally once before epochs
        with self.timing_tracker.time_context("advantage_normalization_time"):
            valid_advantages_tensor = self.rollout_buffer.buffer["advantages"][
                :buffer_actual_size
            ]
            if valid_advantages_tensor.numel() > 0:
                original_shape = valid_advantages_tensor.shape
                flat_adv = valid_advantages_tensor.reshape(-1)
                normalized_flat_adv = (flat_adv - flat_adv.mean()) / (
                    flat_adv.std() + 1e-8
                )
                self.rollout_buffer.buffer["advantages"][:buffer_actual_size] = (
                    normalized_flat_adv.reshape(original_shape)
                )
            else:
                warnings.warn(
                    "No advantages to normalize in BPTT pre-normalization step."
                )

        # Determine boundary-safe start coordinates for sequences (avoid crossing episode boundaries)
        num_possible_starts_per_env = buffer_actual_size - seq_len + 1
        if num_possible_starts_per_env <= 0:
            warnings.warn(
                f"Not enough data in buffer ({buffer_actual_size} steps) to form sequences of length {seq_len}. Skipping BPTT."
            )
            return

        # Use the buffer's boundary-aware sampler to get all valid starts, then apply stride/filtering
        candidate_coords = self.rollout_buffer._sample_sequence_start_indices(
            seq_len=seq_len, batch_size=None
        )

        all_start_coords: list[tuple[int, int]] = []
        if self.bptt_sequence_type == BPTTSequenceType.CHUNKED:
            all_start_coords = [
                (env_idx, t_idx)
                for (env_idx, t_idx) in candidate_coords
                if (t_idx % seq_len) == 0
            ]
        elif self.bptt_sequence_type == BPTTSequenceType.MAXIMUM:
            all_start_coords = list(candidate_coords)
        elif self.bptt_sequence_type == BPTTSequenceType.FIFTY_PERCENT_OVERLAP:
            step_size = max(1, seq_len // 2)
            all_start_coords = [
                (env_idx, t_idx)
                for (env_idx, t_idx) in candidate_coords
                if (t_idx % step_size) == 0
            ]
        else:
            raise ValueError(f"Unknown BPTTSequenceType: {self.bptt_sequence_type}")

        # Release candidate list as soon as it's no longer needed
        del candidate_coords

        if not all_start_coords:
            warnings.warn("No BPTT sequences to sample. Skipping learning.")
            return

        sequences_per_minibatch = (
            self.batch_size
        )  # Here, batch_size means number of sequences per minibatch

        for epoch in range(self.update_epochs):
            np.random.shuffle(all_start_coords)
            num_minibatches_this_epoch = 0
            # Accumulate metrics per epoch to reduce .item() calls
            sum_total_loss = torch.zeros((), device=self.device)
            sum_policy_loss = torch.zeros((), device=self.device)
            sum_value_loss = torch.zeros((), device=self.device)
            sum_entropy_loss = torch.zeros((), device=self.device)
            sum_approx_kl = torch.zeros((), device=self.device)
            sum_clip_fraction = torch.zeros((), device=self.device)
            sum_ev = torch.zeros((), device=self.device)
            sum_actor_norm = torch.zeros((), device=self.device)
            sum_critic_norm = torch.zeros((), device=self.device)

            for i in range(0, len(all_start_coords), sequences_per_minibatch):
                current_coords_minibatch = all_start_coords[
                    i : i + sequences_per_minibatch
                ]
                if not current_coords_minibatch:
                    continue

                # Fetch minibatch of sequences; returns TensorDict on CPU (we'll move per-minibatch tensors to device)
                # Batch_size: [len(current_coords_minibatch), seq_len]
                # "initial_hidden_states" is a non-tensor entry in TD: Dict[str, Tensor(batch_seq_size, layers, size)]
                with self.timing_tracker.time_context("get_sequences_batch_time"):
                    current_minibatch_td = (
                        self.rollout_buffer.get_specific_sequences_tensor_batch(
                            seq_len=seq_len,
                            sequence_coords=current_coords_minibatch,
                            device=self.device,  # Fetch directly on GPU to avoid double transfer
                            include_keys=[
                                "observations",
                                "actions",
                                "log_probs",
                                "advantages",
                                "returns",
                                "values",
                                "action_masks",
                            ],
                            as_plain_dict=True,
                        )
                    )

                if (not isinstance(current_minibatch_td, dict)) or (
                    "observations" not in current_minibatch_td
                ):
                    warnings.warn("Skipping empty or invalid minibatch of sequences.")
                    continue

                # Data is already on device from get_specific_sequences_tensor_batch
                mb_obs_seq = current_minibatch_td["observations"]
                mb_actions_seq = current_minibatch_td["actions"]
                mb_old_log_probs_seq = current_minibatch_td["log_probs"]
                mb_advantages_seq = current_minibatch_td["advantages"]
                mb_returns_seq = current_minibatch_td["returns"]
                mb_old_values_seq = current_minibatch_td["values"]
                mb_action_masks_seq = current_minibatch_td.get("action_masks", None)

                mb_initial_hidden_states_dict = current_minibatch_td.get(
                    "initial_hidden_states", None
                )

                # Free the container as early as possible
                del current_minibatch_td

                current_step_hidden_state_actor = (
                    None  # For actor: {key: (layers, batch_seq_size, hidden_size)}
                )

                if self.recurrent and mb_initial_hidden_states_dict is not None:
                    current_step_hidden_state_actor = {
                        # val is (batch_seq_size, layers, size), permute to (layers, batch_seq_size, size)
                        # Detach to prevent old computation graphs from accumulating
                        # Data is already on device from get_specific_sequences_tensor_batch
                        key: val.permute(1, 0, 2).contiguous().detach()
                        for key, val in mb_initial_hidden_states_dict.items()
                    }

                # Release raw initial hidden state dict once converted
                if mb_initial_hidden_states_dict is not None:
                    mb_initial_hidden_states_dict = None

                with self.timing_tracker.time_context("bptt_loss_calculation_time"):
                    loss_dict = self.compute_loss(
                        mb_obs_seq,
                        mb_actions_seq,
                        mb_old_log_probs_seq,
                        mb_advantages_seq,
                        mb_returns_seq,
                        hidden_state=current_step_hidden_state_actor,
                        old_values=mb_old_values_seq,
                        learn_by_bptt=True,
                        seq_len=seq_len,
                        action_mask=mb_action_masks_seq,
                    )
                loss = loss_dict["loss"]

                # Accumulate metrics (tensor scalars)
                sum_policy_loss += loss_dict["policy_loss"].detach()
                sum_value_loss += loss_dict["value_loss"].detach()
                sum_entropy_loss += loss_dict["entropy_loss"].detach()
                sum_approx_kl += loss_dict["approx_kl"].detach()
                sum_clip_fraction += loss_dict["clip_fraction"].detach()

                # EV using old values vs returns (avoid extra forward)
                with torch.no_grad():
                    ev = self._explained_variance(
                        mb_old_values_seq.reshape(-1), mb_returns_seq.reshape(-1)
                    )
                    sum_ev += ev

                # Free masks before backward
                if mb_action_masks_seq is not None:
                    del mb_action_masks_seq

                with self.timing_tracker.time_context("bptt_backward_pass_time"):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()  # Gradients accumulate over the sequence within this backward call
                    actor_norm = clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    critic_norm = clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                # Track only counts; add aggregated metrics after loop
                sum_total_loss += loss.detach()
                sum_actor_norm += actor_norm.detach()
                sum_critic_norm += critic_norm.detach()

                num_minibatches_this_epoch += 1

                # Clean up BPTT minibatch tensors to free memory
                del mb_obs_seq, mb_actions_seq, mb_old_log_probs_seq
                del mb_advantages_seq, mb_returns_seq, mb_old_values_seq
                if current_step_hidden_state_actor is not None:
                    del current_step_hidden_state_actor

                del loss_dict, loss

                # Check KL divergence for early stopping - only check every 4 minibatches to reduce GPU sync overhead
                # This is a performance optimization that slightly delays early stopping detection
                should_stop = False
                if self.target_kl is not None and num_minibatches_this_epoch % 4 == 0:
                    # Use accumulated KL average instead of single minibatch
                    avg_approx_kl = (sum_approx_kl / max(1, num_minibatches_this_epoch)).item()
                    should_stop = avg_approx_kl > self.target_kl
                    if should_stop:
                        warnings.warn(
                            f"Minibatch: KL divergence {avg_approx_kl:.4f} exceeded target {self.target_kl}. Stopping update for this epoch."
                        )
                        break  # Break from minibatch loop for this epoch

            # Log averaged metrics once per epoch - batch all GPU->CPU transfers into one call
            denom = max(1, num_minibatches_this_epoch)
            inv = 1.0 / denom
            # Stack all metrics into a single tensor for one CPU transfer
            metrics_tensor = torch.stack([
                sum_total_loss * inv,
                sum_policy_loss * inv,
                sum_value_loss * inv,
                sum_entropy_loss * inv,
                sum_approx_kl * inv,
                sum_clip_fraction * inv,
                sum_ev * inv,
                sum_actor_norm * inv,
                sum_critic_norm * inv,
            ]).detach().cpu()

            self.learn_metrics.add("total_loss", float(metrics_tensor[0]))
            self.learn_metrics.add("policy_loss", float(metrics_tensor[1]))
            self.learn_metrics.add("value_loss", float(metrics_tensor[2]))
            self.learn_metrics.add("entropy_loss", float(metrics_tensor[3]))
            self.learn_metrics.add("approx_kl", float(metrics_tensor[4]))
            self.learn_metrics.add("clip_fraction", float(metrics_tensor[5]))
            self.learn_metrics.add("explained_variance", float(metrics_tensor[6]))
            self.learn_metrics.add("actor_grad_norm", float(metrics_tensor[7]))
            self.learn_metrics.add("critic_grad_norm", float(metrics_tensor[8]))

    def add_collection_time(self, collection_time: float) -> None:
        """Add collection time to metrics tracker.

        :param collection_time: Time spent collecting experiences
        :type collection_time: float
        """
        self.last_collection_time = collection_time
        self.total_collection_time += collection_time

    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
        vectorized: bool = True,
        deterministic: bool = True,
        callback: Optional[Callable[[float, Dict[str, float]], None]] = None,
        eval_sampling: Optional[str] = None,
        num_envs_test: Optional[int] = None,
    ) -> float:
        """Returns mean test score of agent in environment.

        :param env: The environment to be tested in
        :type env: GymEnvType
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :param vectorized: Whether the environment is vectorized, defaults to True
        :type vectorized: bool, optional
        :param deterministic: Boolean specifying whether to desired action is stochastic or deterministic, defaults to True
        :type deterministic: bool, optional
        :param callback: Optional callback function that takes the sum of rewards and the last info dictionary as input, defaults to None
        :type callback: Optional[Callable[[float, Dict[str, float]], None]]

        :return: Mean test score of agent in environment
        :rtype: float
        """
        # Override deterministic based on eval_sampling if provided
        # This ensures eval matches training behavior when eval_sampling="distribution"
        if eval_sampling is not None:
            deterministic = eval_sampling == "deterministic"

        # set to evaluation mode. This is important for batch norm and dropout layers
        self.actor.eval()
        self.critic.eval()
        self.set_training_mode(False)

        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") and vectorized else 1

            for _ in range(loop):
                obs, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs, dtype=bool)
                step = 0
                test_hidden_state = (
                    self.get_initial_hidden_state(num_envs) if self.recurrent else None
                )

                last_infos = (
                    [{}] * num_envs if vectorized else {}
                )  # Initialize last_info holder

                while not np.all(finished):
                    if swap_channels:
                        obs = obs_channels_to_first(obs)

                    # Process action mask
                    action_mask = None
                    if vectorized:
                        # Check if info is a list/array of dicts
                        if (
                            isinstance(info, (list, np.ndarray))
                            and len(info) == num_envs
                            and all(isinstance(i, dict) for i in info)
                        ):
                            masks = [env_info.get("action_mask") for env_info in info]
                            # If all environments returned a mask and they are not None
                            if all(m is not None for m in masks):
                                try:
                                    action_mask = np.stack(masks)
                                except Exception as e:
                                    warnings.warn(f"Could not stack action masks: {e}")
                                    action_mask = None
                            # If only some environments returned masks, we probably can't use them reliably
                            elif any(m is not None for m in masks):
                                warnings.warn(
                                    "Action masks not provided for all vectorized environments. Skipping mask."
                                )
                                action_mask = None
                        # Handle case where info might be a single dict even if vectorized (e.g. VecNormalize)
                        elif isinstance(info, dict):
                            action_mask = info.get("action_mask", None)

                    else:  # Not vectorized
                        if isinstance(info, dict):
                            action_mask = info.get("action_mask", None)

                    # Get action
                    if self.recurrent:
                        action, _, _, _, test_hidden_state = self.get_action(
                            obs,
                            action_mask=action_mask,
                            hidden_state=test_hidden_state,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.get_action(
                            obs, action_mask=action_mask, deterministic=deterministic
                        )

                    # Environment step
                    if vectorized:
                        obs, reward, done, trunc, info = env.step(action)
                        last_infos = info  # Store the array of infos
                    else:
                        obs, reward, done, trunc, info_single = env.step(action[0])
                        # Store info in a dictionary for consistency if not vectorized
                        info = {"final_info": info_single} if done or trunc else {}
                        last_infos = info  # Store the single info dict

                    step += 1
                    # Apply same reward normalization as training for fair comparison
                    # but WITHOUT updating running stats (to avoid polluting training statistics)
                    reward_arr = np.array(reward, dtype=np.float32)
                    if (
                        hasattr(self, "normalize_rewards")
                        and self.normalize_rewards
                        and hasattr(self, "reward_rms")
                    ):
                        flat = reward_arr.reshape(-1)
                        std = math.sqrt(self.reward_rms.var.item() + 1e-8)
                        mean = self.reward_rms.mean.item()
                        if std > 0.0:
                            normalized_flat = (flat - mean) / std
                        else:
                            normalized_flat = flat - mean
                        reward_arr = normalized_flat.reshape(reward_arr.shape).astype(
                            np.float32
                        )
                    scores += reward_arr

                    # Check for episode termination
                    newly_finished = (
                        np.logical_or(
                            np.logical_or(done, trunc),
                            (max_steps is not None and step == max_steps),
                        )
                        & ~finished
                    )

                    # Reset hidden state for newly finished environments
                    if self.recurrent and np.any(newly_finished):
                        initial_hidden_states_for_reset = self.get_initial_hidden_state(
                            num_envs
                        )
                        if isinstance(test_hidden_state, dict):
                            mask_t = torch.as_tensor(
                                newly_finished, dtype=torch.bool, device=self.device
                            )
                            for key in test_hidden_state:
                                reset_states = initial_hidden_states_for_reset[key][
                                    :, mask_t, :
                                ]
                                if reset_states.shape[1] > 0:
                                    test_hidden_state[key][:, mask_t, :] = reset_states

                    if np.any(newly_finished):
                        completed_episode_scores[newly_finished] = scores[
                            newly_finished
                        ]
                        finished[newly_finished] = True

                # End of episode loop for one test run
                loop_reward_sum = np.sum(completed_episode_scores)

                # Prepare info for callback
                final_info_for_callback = {}
                if vectorized:
                    if (
                        isinstance(last_infos, (list, np.ndarray))
                        and len(last_infos) > 0
                    ):
                        final_info_for_callback = (
                            last_infos[0] if isinstance(last_infos[0], dict) else {}
                        )
                    elif isinstance(last_infos, dict):
                        final_info_for_callback = last_infos
                else:  # Not vectorized
                    if isinstance(last_infos, dict):
                        final_info_for_callback = last_infos

                if callback is not None:
                    callback(loop_reward_sum, final_info_for_callback)

                rewards.append(np.mean(completed_episode_scores))

        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)

        # cleanup evaluation mode back into the default training mode (e.g. batch norm and dropout layers)
        self.set_training_mode(True)
        self.actor.train()
        self.critic.train()

        return mean_fit
