from typing import Dict, List, Optional, Protocol, Tuple, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical, Distribution, Normal

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.typing import ArrayOrTensor, DeviceType, NetConfigType

DistributionType = Union[Distribution, List[Distribution]]


def sum_independent_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Sum the values of a tensor across the independent dimensions. Assume
    dim=1 if the tensor has more than 1 dimension.

    :param tensor: Tensor to sum.
    :type tensor: torch.Tensor
    :return: Sum of the tensor.
    :rtype: torch.Tensor
    """
    return tensor.sum(dim=1) if len(tensor.shape) > 1 else tensor


def apply_action_mask_discrete(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Apply a mask to the logits.

    :param logits: Logits.
    :type logits: torch.Tensor
    :param mask: Mask.
    :type mask: torch.Tensor
    :return: Logits with mask applied.
    :rtype: torch.Tensor
    """
    # Use masked_fill instead of torch.where + torch.full_like
    # This avoids creating a full tensor of -1e8 values
    return logits.masked_fill(~mask, -1e8)


class DistributionHandler(Protocol):
    """Protocol for distribution handlers that implement sampling, log_prob, and entropy methods."""

    def sample(self, distribution: DistributionType) -> torch.Tensor:
        """Sample an action from the distribution."""
        ...

    def log_prob(
        self, distribution: DistributionType, action: torch.Tensor
    ) -> torch.Tensor:
        """Get the log probability of the action."""
        ...

    def entropy(self, distribution: DistributionType) -> Optional[torch.Tensor]:
        """Get the entropy of the action distribution."""
        ...

    def mode(self, distribution: DistributionType) -> torch.Tensor:
        """Get the mode of the distribution."""
        ...


class NormalHandler:
    """Handler for Normal distributions."""

    def sample(self, distribution: Normal) -> torch.Tensor:
        """Sample an action from the distribution using reparameterization trick.

        :param distribution: Distribution to sample from.
        :type distribution: Normal
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return distribution.sample()

    def log_prob(self, distribution: Normal, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: Distribution to compute log probability for.
        :type distribution: Normal
        :param action: Action.
        :type action: torch.Tensor
        """
        return sum_independent_tensor(distribution.log_prob(action))

    def entropy(self, distribution: Normal) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: Distribution to compute entropy for.
        :type distribution: Normal
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return sum_independent_tensor(distribution.entropy())

    def mode(self, distribution: Normal) -> torch.Tensor:
        """Get the mode of the distribution.

        :param distribution: Distribution to compute mode for.
        :type distribution: Normal
        :return: Mode of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.mean


class BernoulliHandler:
    """Handler for Bernoulli distributions."""

    def sample(self, distribution: Bernoulli) -> torch.Tensor:
        """Sample an action from the distribution.

        :param distribution: Distribution to sample from.
        :type distribution: Bernoulli
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return distribution.sample()

    def log_prob(self, distribution: Bernoulli, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: Distribution to compute log probability for.
        :type distribution: Bernoulli
        :param action: Action.
        :type action: torch.Tensor
        """
        return distribution.log_prob(action).sum(dim=1)

    def entropy(self, distribution: Bernoulli) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: Distribution to compute entropy for.
        :type distribution: Bernoulli
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.entropy().sum(dim=1)

    def mode(self, distribution: Bernoulli) -> torch.Tensor:
        """Get the mode of the distribution.

        :param distribution: Distribution to compute mode for.
        :type distribution: Bernoulli
        :return: Mode of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.mode


class CategoricalHandler:
    """Handler for Categorical distributions."""

    def sample(self, distribution: Categorical) -> torch.Tensor:
        """Sample an action from the distribution.

        :param distribution: Distribution to sample from.
        :type distribution: Categorical
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return distribution.sample()

    def log_prob(self, distribution: Categorical, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: Distribution to compute log probability for.
        :type distribution: Categorical
        :param action: Action.
        :type action: torch.Tensor
        """
        return distribution.log_prob(action.long())

    def entropy(self, distribution: Categorical) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: Distribution to compute entropy for.
        :type distribution: Categorical
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.entropy()

    def mode(self, distribution: Categorical) -> torch.Tensor:
        """Get the mode of the distribution.

        :param distribution: Distribution to compute mode for.
        :type distribution: Categorical
        :return: Mode of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.mode


class MultiCategoricalHandler:
    """Handler for list of Categorical distributions (MultiDiscrete action spaces)."""

    def sample(self, distribution: List[Categorical]) -> torch.Tensor:
        """Sample an action from the distribution.

        :param distribution: List of Categorical distributions to sample from.
        :type distribution: List[Categorical]
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return torch.stack([dist.sample() for dist in distribution], dim=1)

    def log_prob(
        self, distribution: List[Categorical], action: torch.Tensor
    ) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: List of Categorical distributions to compute log probability for.
        :type distribution: List[Categorical]
        :param action: Action.
        :type action: torch.Tensor
        """
        unbinded_actions = torch.unbind(action.long(), dim=1)
        multi_log_prob = [
            dist.log_prob(act) for dist, act in zip(distribution, unbinded_actions)
        ]
        return torch.stack(multi_log_prob, dim=1).sum(dim=1)

    def entropy(self, distribution: List[Categorical]) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: List of Categorical distributions to compute entropy for.
        :type distribution: List[Categorical]
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return torch.stack([dist.entropy() for dist in distribution], dim=1).sum(dim=1)

    def mode(self, distribution: List[Categorical]) -> torch.Tensor:
        """Get the mode of the distribution.

        :param distribution: List of Categorical distributions to compute mode for.
        :type distribution: List[Categorical]
        :return: Mode of the action distribution.
        :rtype: torch.Tensor
        """
        return torch.stack([dist.mode for dist in distribution], dim=1)


class HybridHandler:
    """Handler for hybrid distributions: (Categorical for discrete, Normal for continuous).

    Expects a tuple of (Categorical, Normal) distributions.
    The returned action is concatenated as [discrete_index, continuous...].
    """

    def sample(self, distribution: Tuple[Categorical, Normal]) -> torch.Tensor:
        cat_dist, cont_dist = distribution
        discrete = cat_dist.sample().unsqueeze(1).float()
        continuous = cont_dist.sample()
        return torch.cat([discrete, continuous], dim=1)

    def log_prob(
        self, distribution: Tuple[Categorical, Normal], action: torch.Tensor
    ) -> torch.Tensor:
        cat_dist, cont_dist = distribution
        discrete = action[:, 0]
        continuous = action[:, 1:]
        return cat_dist.log_prob(discrete.long()) + sum_independent_tensor(
            cont_dist.log_prob(continuous)
        )

    def entropy(self, distribution: Tuple[Categorical, Normal]) -> torch.Tensor:
        cat_dist, cont_dist = distribution
        return cat_dist.entropy() + sum_independent_tensor(cont_dist.entropy())

    def mode(self, distribution: Tuple[Categorical, Normal]) -> torch.Tensor:
        # Fallback: use argmax for categorical and mean for normal
        cat_dist, cont_dist = distribution
        discrete = cat_dist.mode.unsqueeze(1).float()
        continuous = cont_dist.mean
        return torch.cat([discrete, continuous], dim=1)


class TorchDistribution:
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param distribution: Distribution to wrap.
    :type distribution: Union[Distribution, List[Distribution]]
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    """

    # Map distribution types to their handlers
    _handlers: Dict[Type, DistributionHandler] = {
        Normal: NormalHandler(),
        Bernoulli: BernoulliHandler(),
        Categorical: CategoricalHandler(),
        list: MultiCategoricalHandler(),
    }

    def __init__(
        self,
        distribution: DistributionType,
        squash_output: bool = False,
    ) -> None:
        if isinstance(distribution, list):
            assert all(
                isinstance(d, Categorical) for d in distribution
            ), "Only list of Categorical distributions are supported (for MultiDiscrete action spaces)."

        self.distribution = distribution
        self.squash_output = squash_output
        self.sampled_action = None
        self._handler = self._get_handler(distribution)

    def _get_handler(self, distribution: DistributionType) -> DistributionHandler:
        """Get the appropriate handler for the distribution type.

        :param distribution: Distribution to get handler for.
        :type distribution: DistributionType
        :return: Appropriate handler for the distribution type.
        :rtype: DistributionHandler
        """
        if isinstance(distribution, list):
            return self._handlers[list]
        # Hybrid: tuple of (Categorical, Normal)
        if isinstance(distribution, tuple) and len(distribution) == 2:
            if isinstance(distribution[0], Categorical) and isinstance(
                distribution[1], Normal
            ):
                return HybridHandler()

        for dist_type, handler in self._handlers.items():
            if isinstance(distribution, dist_type) and dist_type is not list:
                return handler

        raise NotImplementedError(f"Distribution {type(distribution)} not supported.")

    def sample(self) -> torch.Tensor:
        """Sample an action from the distribution.

        :return: Action from the distribution.
        :rtype: torch.Tensor
        """
        self.sampled_action = self._handler.sample(self.distribution)

        if self.squash_output:
            return torch.tanh(self.sampled_action)

        return self.sampled_action

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        if not self.squash_output:
            return self._handler.log_prob(self.distribution, action)

        # For squashed, ensure we have a pre-tanh value
        pre_tanh = self.sampled_action
        if pre_tanh is None:
            eps = 1e-6
            pre_tanh = torch.atanh(action.clamp(-1 + eps, 1 - eps))
        log_prob = self._handler.log_prob(self.distribution, pre_tanh)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1)
        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor, None
        """
        # No analytical form for entropy with squashed outputs so must
        # use -log_prob.mean() in algorithm instead
        if self.squash_output:
            return None

        return self._handler.entropy(self.distribution)

    def mode(self) -> torch.Tensor:
        """Get the mode of the distribution.

        :return: Mode of the distribution.
        :rtype: torch.Tensor
        """
        action = self._handler.mode(self.distribution)
        if self.squash_output:
            action = torch.tanh(action)
        return action


class EvolvableDistribution(EvolvableWrapper):
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param network: Network that outputs the logits of the distribution.
    :type network: EvolvableModule
    :param action_std_init: Initial log standard deviation of the action distribution. Defaults to 0.0.
    :type action_std_init: float
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    :param device: Device to use for the network.
    :type device: DeviceType
    """

    wrapped: EvolvableModule
    dist: Optional[TorchDistribution]
    mask: Optional[ArrayOrTensor]
    log_std: Optional[torch.nn.Parameter]

    def __init__(
        self,
        action_space: spaces.Space,
        network: EvolvableModule,
        action_std_init: float = 0.0,
        squash_output: bool = False,
        device: DeviceType = "cpu",
    ):
        super().__init__(network)

        self.action_space = action_space
        self.action_dim = spaces.flatdim(action_space)
        self.action_std_init = action_std_init
        self.device = device
        self.squash_output = squash_output and isinstance(action_space, spaces.Box)
        self.dist = None
        self.mask = None

        # For continuous action spaces, we also learn the standard
        # deviation (log_std) of the action distribution
        if isinstance(action_space, spaces.Box):
            self.log_std = torch.nn.Parameter(
                torch.ones(1, np.prod(action_space.shape), device=device)
                * action_std_init
            )
        elif isinstance(action_space, spaces.Tuple):
            # Assume (Discrete(n), Box(k,)) hybrid
            assert (
                len(action_space.spaces) == 2
                and isinstance(action_space.spaces[0], spaces.Discrete)
                and isinstance(action_space.spaces[1], spaces.Box)
            ), "Tuple action space must be (Discrete, Box)."
            k = int(np.prod(action_space.spaces[1].shape))
            self.log_std = torch.nn.Parameter(
                torch.ones(1, k, device=device) * action_std_init
            )

    @property
    def net_config(self) -> NetConfigType:
        """Configuration of the network.

        :return: Configuration of the network.
        :rtype: NetConfigType
        """
        return self.wrapped.net_config

    def get_distribution(self, logits: torch.Tensor) -> TorchDistribution:
        """Get the distribution over the action space given an observation.

        :param logits: Output of the network, either logits or probabilities.
        :type logits: torch.Tensor
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        # Normal distribution for Continuous action spaces
        if isinstance(self.action_space, spaces.Box):
            log_std = self.log_std.expand_as(logits)
            action_std = torch.exp(log_std)
            dist = Normal(loc=logits, scale=action_std)

        # Categorical distribution for Discrete action spaces
        elif isinstance(self.action_space, spaces.Discrete):
            dist = Categorical(logits=logits)

        # List of categorical distributions for MultiDiscrete action spaces
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            dist = [
                Categorical(logits=split)
                for split in torch.split(logits, list(self.action_space.nvec), dim=1)
            ]

        # Bernoulli distribution for MultiBinary action spaces
        elif isinstance(self.action_space, spaces.MultiBinary):
            dist = Bernoulli(logits=logits)
        elif isinstance(self.action_space, spaces.Tuple):
            # Hybrid: split logits into discrete and continuous means
            disc_space: spaces.Discrete = self.action_space.spaces[0]  # type: ignore[assignment]
            box_space: spaces.Box = self.action_space.spaces[1]  # type: ignore[assignment]
            n = disc_space.n
            k = int(np.prod(box_space.shape))
            assert logits.shape[1] == n + k, "Hybrid logits must have n+k outputs"
            disc_logits = logits[:, :n]
            cont_means = logits[:, n:]
            log_std = self.log_std.expand_as(cont_means)
            action_std = torch.exp(log_std)
            cat = Categorical(logits=disc_logits)
            norm = Normal(loc=cont_means, scale=action_std)
            dist = (cat, norm)
        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )

        return TorchDistribution(dist, self.squash_output)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        if self.dist is None:
            raise ValueError("Distribution not initialized. Call forward first.")

        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        if self.dist is None:
            raise ValueError("Distribution not initialized. Call forward first.")

        return self.dist.entropy()

    def apply_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a mask to the logits.

        :param logits: Logits (already on device).
        :type logits: torch.Tensor
        :param mask: Mask (already converted to tensor and reshaped).
        :type mask: torch.Tensor
        :return: Logits with mask applied.
        :rtype: torch.Tensor
        """
        # Vectorized masked_fill works for all discrete action spaces
        # No need to split for MultiDiscrete - masked_fill is element-wise
        if isinstance(self.action_space, spaces.Tuple):
            # Only mask the discrete part (first n logits)
            disc_space: spaces.Discrete = self.action_space.spaces[0]  # type: ignore[assignment]
            n = disc_space.n
            disc_logits = logits[:, :n]
            cont_logits = logits[:, n:]
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand_as(disc_logits)
            masked_disc = disc_logits.masked_fill(~mask, -1e8)
            return torch.cat([masked_disc, cont_logits], dim=1)
        else:
            # Discrete, MultiDiscrete, MultiBinary - single vectorized operation
            return logits.masked_fill(~mask, -1e8)

    def build_dist_from_latent(self, latent, action_mask=None):
        logits = self.wrapped(latent)
        if action_mask is not None:
            logits = self.apply_mask(logits, action_mask)
        return self.get_distribution(logits)

    def log_prob_from_latent(self, latent, actions, action_mask=None):
        dist = self.build_dist_from_latent(latent, action_mask)
        return dist.log_prob(actions)

    def entropy_from_latent(self, latent, action_mask=None):
        dist = self.build_dist_from_latent(latent, action_mask)
        return dist.entropy()

    def forward(
        self,
        latent: torch.Tensor,
        action_mask: Optional[ArrayOrTensor] = None,
        sample: bool = True,
        deterministic: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[None, None, torch.Tensor]
    ]:
        """Forward pass of the network.

        :param latent: Latent space representation.
        :type latent: torch.Tensor
        :param action_mask: Mask to apply to the logits. Defaults to None.
        :type action_mask: Optional[ArrayOrTensor]
        :param sample: Whether to sample an action from the distribution. Defaults to True.
        :type sample: bool, optional
        :param deterministic: Whether to return a deterministic action. Defaults to False.
        :type deterministic: bool, optional
        :return: Action, log probability of the action, and entropy of the distribution.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        logits = self.wrapped(latent)

        if action_mask is not None:
            # Fast path: mask is already a bool tensor on correct device
            if (
                isinstance(action_mask, torch.Tensor)
                and action_mask.dtype == torch.bool
            ):
                if action_mask.device != logits.device:
                    action_mask = action_mask.to(logits.device, non_blocking=True)
                mask_tensor = action_mask.view(logits.shape)
            else:
                # Slow path: convert from numpy/list
                if isinstance(action_mask, (np.ndarray, list)):
                    action_mask = (
                        np.stack(action_mask)
                        if (
                            hasattr(action_mask, "dtype")
                            and action_mask.dtype == np.object_
                        )
                        or isinstance(action_mask, list)
                        else action_mask
                    )
                mask_tensor = torch.as_tensor(
                    action_mask, dtype=torch.bool, device=self.device
                ).view(logits.shape)

            # Single vectorized masked_fill
            logits = logits.masked_fill(~mask_tensor, -1e8)

        # Distribution from logits
        self.dist = self.get_distribution(logits)

        action = None
        log_prob = None

        if deterministic:
            action = self.dist.mode()
            log_prob = self.dist.log_prob(action)
        elif sample:
            action = self.dist.sample()
            log_prob = self.dist.log_prob(action)

        entropy = self.dist.entropy()
        return action, log_prob, entropy

    def clone(self) -> "EvolvableDistribution":
        """Clones the distribution.

        :return: Cloned distribution.
        :rtype: EvolvableDistribution
        """
        clone = EvolvableDistribution(
            action_space=self.action_space,
            network=self.wrapped.clone(),
            action_std_init=self.action_std_init,
            squash_output=self.squash_output,
            device=self.device,
        )
        clone.rng = self.rng
        return clone
