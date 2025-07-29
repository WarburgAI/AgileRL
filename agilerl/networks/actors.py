import os
from typing import Optional, Tuple, Type, Union

import dotenv
import torch
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.distributions import EvolvableDistribution
from agilerl.typing import ArrayOrTensor, NetConfigType, TorchObsType

dotenv.load_dotenv()

if os.getenv("USE_EXPERIMENTAL_DISTRIBUTIONS", "False") == "True":
    from agilerl.networks.distributions_experimental import EvolvableDistribution
else:
    from agilerl.networks.distributions import EvolvableDistribution


class DeterministicActor(EvolvableNetwork):
    """Deterministic actor network for policy-gradient algorithms. Given an observation,
    it outputs the mean of the action distribution. This is useful for e.g. DDPG, SAC, TD3.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: Union[spaces.Box, spaces.Discrete]
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: NetConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[NetConfigType]
    :param clip_actions: Whether to clip the actions to the action space.
    :type clip_actions: bool
    :param min_latent_dim: Minimum dimension of the latent space representation.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation.
    :type max_latent_dim: int
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to use for the network.
    :type device: str
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: Optional[int]
    :param encoder_name: Name of the encoder network.
    :type encoder_name: str
    """

    supported_spaces = (spaces.Box, spaces.Discrete)

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: Union[spaces.Box, spaces.Discrete],
        encoder_cls: Optional[Union[str, Type[EvolvableModule]]] = None,
        encoder_config: Optional[NetConfigType] = None,
        head_config: Optional[NetConfigType] = None,
        clip_actions: bool = True,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        latent_dim: int = 32,
        simba: bool = False,
        recurrent: bool = False,
        device: str = "cpu",
        random_seed: Optional[int] = None,
        encoder_name: str = "encoder",
    ):
        super().__init__(
            observation_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            latent_dim=latent_dim,
            simba=simba,
            recurrent=recurrent,
            device=device,
            random_seed=random_seed,
            encoder_name=encoder_name,
        )

        self.clip_actions = clip_actions
        if isinstance(action_space, spaces.Box):
            self.action_low = torch.as_tensor(action_space.low, device=self.device)
            self.action_high = torch.as_tensor(action_space.high, device=self.device)
        else:
            self.action_low = None
            self.action_high = None

        # Set output activation based on action space
        if head_config is not None and "output_activation" in head_config:
            output_activation = head_config["output_activation"]
        elif isinstance(action_space, spaces.Box):
            # Squash output by default if continuous action space
            output_activation = "Tanh"
        elif isinstance(action_space, spaces.Discrete):
            output_activation = "Softmax"
        else:
            output_activation = None

        if head_config is None:
            head_config = MlpNetConfig(
                hidden_size=[32], output_activation=output_activation
            )
        else:
            head_config["output_activation"] = output_activation

        self.build_network_head(head_config)
        self.output_activation = head_config.get("output_activation", output_activation)

    @torch.compiler.disable
    @staticmethod
    def rescale_action(
        action: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        output_activation: Optional[str] = None,
    ) -> torch.Tensor:
        """Rescale an action to the original action space.

        :param action: Action.
        :type action: torch.Tensor
        :param low: Minimum action.
        :type low: torch.Tensor
        :param high: Maximum action.
        :type high: torch.Tensor
        :param output_activation: Output activation function.
        :type output_activation: Optional[str]
        :return: Rescaled action.
        :rtype: torch.Tensor
        """
        if output_activation in ["Tanh", "Softsign"]:
            prescaled_min, prescaled_max = -1.0, 1.0
        elif output_activation in ["Sigmoid", "Softmax", "GumbelSoftmax"]:
            prescaled_min, prescaled_max = 0.0, 1.0
        else:
            # For unbounded network outputs, we just return the action
            return action

        # If the action space is unbounded, we just return the action
        if low.isinf().any() or high.isinf().any():
            rescaled_action = action
        else:
            rescaled_action = low + (high - low) * (action - prescaled_min) / (
                prescaled_max - prescaled_min
            )

        return rescaled_action

    def build_network_head(self, net_config: Optional[NetConfigType] = None) -> None:
        """Builds the head of the network.

        :param net_config: Configuration of the head.
        :type net_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=net_config,
        )

    def forward(self, obs: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.extract_features(obs)
        action = self.head_net(latent)

        # Action scaling only relevant for continuous action spaces
        if isinstance(self.action_space, spaces.Box) and self.clip_actions:
            action = DeterministicActor.rescale_action(
                action=action,
                low=self.action_low,
                high=self.action_high,
                output_activation=self.output_activation,
            )

        return action

    def recreate_network(self) -> None:
        """Recreates the network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)


class StochasticActor(EvolvableNetwork):
    """Stochastic actor network for policy-gradient algorithms. Given an observation, constructs
    a distribution over the action space from the logits output by the network. Contains methods
    to sample actions and compute log probabilities and the entropy of the action distribution,
    relevant for many policy-gradient algorithms such as PPO, A2C, TRPO.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Space
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: NetConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[NetConfigType]
    :param action_std_init: Initial log standard deviation of the action distribution. Defaults to 0.0.
    :type action_std_init: float
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    :param min_latent_dim: Minimum dimension of the latent space representation.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation.
    :type max_latent_dim: int
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to use for the network.
    :type device: str
    :param use_experimental_distribution: Whether to use the experimental distribution implementation, which
        includes several optimizations related to using torch primitives for statistics calculations. Defaults to False.
    :type use_experimental_distribution: bool
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: Optional[int]
    :param encoder_name: Name of the encoder network.
    :type encoder_name: str
    """

    head_net: EvolvableDistribution
    supported_spaces = (
        spaces.Box,
        spaces.Discrete,
        spaces.MultiDiscrete,
        spaces.MultiBinary,
    )

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        encoder_cls: Optional[Union[str, Type[EvolvableModule]]] = None,
        encoder_config: Optional[NetConfigType] = None,
        head_config: Optional[NetConfigType] = None,
        action_std_init: float = 0.0,
        squash_output: bool = False,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        latent_dim: int = 32,
        simba: bool = False,
        recurrent: bool = False,
        device: str = "cpu",
        use_experimental_distribution: bool = False,
        random_seed: Optional[int] = None,
        encoder_name: str = "encoder",
        **kwargs,
    ):
        super().__init__(
            observation_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            latent_dim=latent_dim,
            simba=simba,
            recurrent=recurrent,
            device=device,
            random_seed=random_seed,
            encoder_name=encoder_name,
        )

        for key, value in kwargs.items():
            print(
                f"an extra argument has been passed and will be ignored: {key} = {value}"
            )

        # Require the head to output logits to parameterize a distribution
        if head_config is None:
            head_config = MlpNetConfig(hidden_size=[32], output_activation=None)
        else:
            head_config["output_activation"] = None

        self.action_std_init = action_std_init
        self.squash_output = squash_output
        self.action_space = action_space
        self.use_experimental_distribution = use_experimental_distribution

        self.build_network_head(head_config)
        self.output_activation = None

        if isinstance(self.action_space, spaces.Box):
            self.action_low = torch.as_tensor(self.action_space.low, device=self.device)
            self.action_high = torch.as_tensor(
                self.action_space.high, device=self.device
            )
        else:
            self.action_low = None
            self.action_high = None

        # Wrap the network in an EvolvableDistribution
        if use_experimental_distribution:
            from agilerl.networks.distributions_experimental import (
                EvolvableDistribution,
            )
        else:
            from agilerl.networks.distributions import EvolvableDistribution

        self.head_net = EvolvableDistribution(
            action_space=action_space,
            network=self.head_net,
            action_std_init=action_std_init,
            squash_output=squash_output,
            device=device,
        )

    def build_network_head(self, net_config: Optional[NetConfigType] = None) -> None:
        """Builds the head of the network.

        :param net_config: Configuration of the head.
        :type net_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=net_config,
        )

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale the action to the action space.

        :param action: Action.
        :type action: torch.Tensor
        :return: Scaled action.
        :rtype: torch.Tensor
        """
        return self.action_low + (
            0.5 * (action + 1.0) * (self.action_high - self.action_low)
        )

    def forward_head(
        self,
        latent: torch.Tensor,
        action_mask: Optional[ArrayOrTensor] = None,
        sample: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the network head.

        :param latent: Latent space representation.
        :type latent: torch.Tensor
        :param action_mask: Action mask.
        :type action_mask: Optional[ArrayOrTensor]
        :param sample: Whether to sample an action from the distribution. Defaults to True.
        :type sample: bool, optional
        :param deterministic: Whether to return a deterministic action. Defaults to False.
        :type deterministic: bool, optional
        :return: Action, log probability of the action, and entropy of the distribution.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Handle sequence inputs - latent may be (batch, seq_len, latent_dim)
        if len(latent.shape) == 3:
            batch_size, seq_len = latent.shape[0], latent.shape[1]
            # Flatten for distribution processing
            latent_flat = latent.reshape(batch_size * seq_len, -1)

            # Process through distribution head
            action_flat, log_prob_flat, entropy_flat = self.head_net.forward(
                latent_flat, action_mask, sample=sample, deterministic=deterministic
            )

            # Reshape back to sequence format
            action = action_flat.reshape(batch_size, seq_len, -1)
            log_prob = (
                log_prob_flat.reshape(batch_size, seq_len, -1)
                if len(log_prob_flat.shape) > 1
                else log_prob_flat.reshape(batch_size, seq_len)
            )
            entropy = (
                entropy_flat.reshape(batch_size, seq_len, -1)
                if len(entropy_flat.shape) > 1
                else entropy_flat.reshape(batch_size, seq_len)
            )

            return action, log_prob, entropy
        else:
            return self.head_net.forward(
                latent, action_mask, sample=sample, deterministic=deterministic
            )

    def forward(
        self,
        obs: TorchObsType,
        action_mask: Optional[ArrayOrTensor] = None,
        hidden_state: Optional[TorchObsType] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :param action_mask: Action mask.
        :type action_mask: Optional[ArrayOrTensor]
        :param hidden_state: Hidden state for recurrent networks.
        :type hidden_state: Optional[TorchObsType]
        :return: Action, log probability, entropy, and optionally next hidden state.
        :rtype: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        """
        if self.recurrent and hidden_state is not None:
            latent, next_hidden_state = self.extract_features(
                obs, hidden_state=hidden_state
            )
            action, log_prob, entropy = self.forward_head(latent, action_mask)

            # Action scaling only relevant for continuous action spaces with squashing
            if isinstance(self.action_space, spaces.Box) and self.squash_output:
                action = self.scale_action(action)

            return action, log_prob, entropy, next_hidden_state
        else:
            latent = self.extract_features(obs)
            action, log_prob, entropy = self.forward_head(latent, action_mask)

            # Action scaling only relevant for continuous action spaces with squashing
            if isinstance(self.action_space, spaces.Box) and self.squash_output:
                action = self.scale_action(action)

            return action, log_prob, entropy

    def action_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        return self.head_net.log_prob(action)

    def action_entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return self.head_net.entropy()

    def recreate_network(self) -> None:
        """Recreates the network with the same parameters as the current network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=self.head_net.net_config,
        )

        head_net = EvolvableDistribution(
            self.action_space,
            head_net,
            action_std_init=self.action_std_init,
            squash_output=self.squash_output,
            device=self.device,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)
