"""Functions for collecting rollouts for on-policy algorithms."""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from agilerl.algorithms import PPO
from agilerl.networks import StochasticActor
from agilerl.typing import GymEnvType
from agilerl.utils.metrics import TimingTracker

SupportedOnPolicy = PPO


class RolloutHook(ABC):
    """Abstract base class for rollout collection hooks."""

    @abstractmethod
    def can_handle(self, agent) -> bool:
        """Check if this hook can handle the given agent."""
        pass

    def on_rollout_start(
        self, agent, env, n_steps: int, reset_on_collect: bool
    ) -> Dict[str, Any]:
        """Called at the start of rollout collection."""
        return {}

    def on_step_start(
        self, agent, obs, info, step_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Called at the start of each step."""
        return step_data

    def process_action_result(
        self, agent, action_result: Tuple, recurrent: bool
    ) -> Tuple:
        """Process the action result from agent.get_action()."""
        return action_result

    def process_reward(
        self, agent, reward, obs, next_obs, action, step_data: Dict[str, Any]
    ) -> Any:
        """Process and potentially modify the reward."""
        return reward

    def prepare_buffer_data(
        self,
        agent,
        obs,
        action,
        reward,
        done,
        value,
        log_prob,
        next_obs,
        hidden_state,
        step_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare data for buffer addition."""
        return {
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "value": value,
            "log_prob": log_prob,
            "next_obs": next_obs,
            "hidden_state": hidden_state,
            **step_data,
        }

    def add_to_buffer(self, agent, buffer_data: Dict[str, Any]) -> None:
        """Add data to the rollout buffer."""
        # Standard buffer addition
        reward_np = np.atleast_1d(buffer_data["reward"])
        done_np = np.atleast_1d(buffer_data["done"])
        value_np = np.atleast_1d(buffer_data["value"])
        log_prob_np = np.atleast_1d(buffer_data["log_prob"])

        agent.rollout_buffer.add(
            obs=buffer_data["obs"],
            action=buffer_data["action"],
            reward=reward_np,
            done=done_np,
            value=value_np,
            log_prob=log_prob_np,
            next_obs=buffer_data["next_obs"],
            hidden_state=buffer_data["hidden_state"],
            episode_start=done_np,
        )

    def on_episode_end(
        self, agent, env_idx: int, is_terminal: np.ndarray, step_data: Dict[str, Any]
    ) -> None:
        """Called when an episode ends."""
        pass

    def on_rollout_end(self, agent, env, step_data: Dict[str, Any]) -> None:
        """Called at the end of rollout collection."""
        pass


class ICMHook(RolloutHook):
    """Hook for ICM-specific rollout collection logic."""

    def can_handle(self, agent) -> bool:
        return hasattr(agent, "get_intrinsic_reward")

    def on_rollout_start(
        self, agent, env, n_steps: int, reset_on_collect: bool
    ) -> Dict[str, Any]:
        return {
            "encoder_last_output": None,
            "encoder_output": None,
            "last_obs_for_icm": None,
        }

    def process_action_result(
        self, agent, action_result: Tuple, recurrent: bool
    ) -> Tuple:
        # ICM may return encoder output as 6th element
        if len(action_result) == 6:
            return action_result  # ICM with shared encoder
        else:
            # Standard case - add None for encoder output
            return (
                action_result + (None,)
                if len(action_result) == 5
                else action_result + (None, None)
            )

    def process_reward(
        self, agent, reward, obs, next_obs, action, step_data: Dict[str, Any]
    ) -> Any:
        # Update encoder states for ICM
        encoder_output = step_data.get("encoder_output")
        if (
            hasattr(agent, "use_shared_encoder_for_icm")
            and agent.use_shared_encoder_for_icm
        ):
            step_data["encoder_last_output"] = encoder_output
        else:
            step_data["last_obs_for_icm"] = obs

        # Calculate intrinsic reward with timing
        with agent.timing_tracker.time_context("intrinsic_reward_calculation"):
            intrinsic_reward, _, _ = agent.get_intrinsic_reward(
                action_batch=action,
                obs_batch=step_data.get("last_obs_for_icm"),
                next_obs_batch=next_obs,
                embedded_obs=step_data.get("encoder_last_output"),
                embedded_next_obs=encoder_output,
                hidden_state_obs=step_data.get("current_hidden_state_for_buffer"),
                hidden_state_next_obs=(
                    agent.hidden_state if hasattr(agent, "hidden_state") else None
                ),
            )

        # Combine extrinsic and intrinsic rewards
        # ICM already applies intrinsic_reward_weight internally; just add it.
        combined_reward = reward + intrinsic_reward.detach().cpu().numpy()

        return combined_reward

    def add_to_buffer(self, agent, buffer_data: Dict[str, Any]) -> None:
        # ICM uses custom buffer addition
        encoder_output = buffer_data.get("encoder_output")

        encoder_output_np = (
            np.atleast_1d(encoder_output) if encoder_output is not None else None
        )

        agent.add_to_rollout_buffer(
            obs=buffer_data["obs"],
            action=buffer_data["action"],
            reward=np.atleast_1d(buffer_data["reward"]),
            done=np.atleast_1d(buffer_data["done"]),
            value=np.atleast_1d(buffer_data["value"]),
            log_prob=np.atleast_1d(buffer_data["log_prob"]),
            next_obs=buffer_data["next_obs"],
            hidden_state=buffer_data["hidden_state"],
            encoder_out=encoder_output_np,
        )


class CVARHook(RolloutHook):
    """Hook for CVAR-specific rollout collection logic."""

    def can_handle(self, agent) -> bool:
        return hasattr(agent, "_episode_returns")

    def on_rollout_start(
        self, agent, env, n_steps: int, reset_on_collect: bool
    ) -> Dict[str, Any]:
        # Reset CVAR accumulators for next epoch/rollout
        agent._nu_delta_sum = 0.0
        agent._bad_trajectory_count = 0
        agent._total_trajectory_steps = 0
        agent.cvarlam = agent.cvarlam + agent.lam_lr * (agent.cvar_beta - agent.nu)
        return {}

    def on_step_start(
        self, agent, obs, info, step_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Accumulate episode returns
        if "reward" in step_data:  # This will be set by process_reward
            agent._episode_returns += np.asarray(step_data["reward"], dtype=np.float32)
        return step_data

    def prepare_buffer_data(
        self,
        agent,
        obs,
        action,
        reward,
        done,
        value,
        log_prob,
        next_obs,
        hidden_state,
        step_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        # CVAR-specific: accumulate episode returns first
        agent._episode_returns += np.asarray(reward, dtype=np.float32)

        # Ensure arrays are properly shaped for CVAR calculations
        reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
        is_terminal_arr = np.asarray(done, dtype=bool).reshape(-1)
        value_arr = np.asarray(value, dtype=np.float32).reshape(-1)

        # Calculate 'updates' term based on CVAR logic
        is_bad_step_arr = (agent._episode_returns + value_arr - reward_arr) < agent.nu

        # Initialize updates to zeros
        updates = np.zeros_like(reward_arr, dtype=np.float32)
        agent._nu_delta_sum += np.sum(agent._episode_returns + value_arr - reward_arr)

        if np.any(is_bad_step_arr):
            # Calculate potential update only for 'bad' steps
            potential_updates_for_bad_steps = (
                agent.delay
                * agent.cvarlam
                / (1.0 - agent.cvar_alpha + 1e-8)
                * (agent.nu - (agent._episode_returns + value_arr - reward_arr))
            )

            # Apply these potential updates only to the elements where is_bad_step_arr is True
            updates[is_bad_step_arr] = potential_updates_for_bad_steps[is_bad_step_arr]

            # Define the clipping threshold based on the absolute value and cvar_clip_ratio
            _clip_threshold_values = np.abs(value_arr) * agent.cvar_clip_ratio

            # Clip the updates only for the 'bad' steps
            updates[is_bad_step_arr] = np.minimum(
                updates[is_bad_step_arr], _clip_threshold_values[is_bad_step_arr]
            )

        # Update epoch-level accumulators for nu/cvarlam updates
        agent._bad_trajectory_count += np.sum(is_bad_step_arr)
        agent._total_trajectory_steps += value_arr.size

        return {
            "obs": obs,
            "action": action,
            "reward": reward_arr,
            "done": is_terminal_arr,
            "value": value_arr,
            "log_prob": np.asarray(log_prob, dtype=np.float32).reshape(-1),
            "next_obs": next_obs,
            "hidden_state": hidden_state,
            "update": updates,
        }

    def add_to_buffer(self, agent, buffer_data: Dict[str, Any]) -> None:
        # CVAR uses standard buffer but with update terms
        agent.rollout_buffer.add(
            obs=buffer_data["obs"],
            action=buffer_data["action"],
            reward=buffer_data["reward"],
            done=buffer_data["done"],
            value=buffer_data["value"],
            log_prob=buffer_data["log_prob"],
            update=buffer_data["update"],
            next_obs=buffer_data["next_obs"],
            hidden_state=buffer_data["hidden_state"],
        )

    def on_episode_end(
        self, agent, env_idx: int, is_terminal: np.ndarray, step_data: Dict[str, Any]
    ) -> None:
        # Reset episode returns for finished environments
        agent._episode_returns[is_terminal] = 0.0

    def on_rollout_end(self, agent, env, step_data: Dict[str, Any]) -> None:
        # Update nu based on collected stats
        if agent._total_trajectory_steps > 0:
            nu_delta = agent._nu_delta_sum / agent._total_trajectory_steps
            agent.nu = nu_delta * agent.nu_delay


class StandardPPOHook(RolloutHook):
    """Hook for standard PPO rollout collection logic."""

    def can_handle(self, agent) -> bool:
        return True  # Fallback for all agents

    def process_action_result(
        self, agent, action_result: Tuple, recurrent: bool
    ) -> Tuple:
        # Ensure consistent format - add None placeholders for encoder output
        if len(action_result) == 4:  # Non-recurrent
            return action_result + (
                None,
                None,
            )  # (action, log_prob, entropy, value, None, None)
        elif len(action_result) == 5:  # Recurrent
            return action_result + (
                None,
            )  # (action, log_prob, entropy, value, hidden, None)
        else:
            return action_result  # Already has encoder output


def get_rollout_hooks(agent) -> List[RolloutHook]:
    """Get appropriate rollout hooks for the given agent."""
    hooks = []

    # Check for specific algorithm hooks first
    if ICMHook().can_handle(agent):
        hooks.append(ICMHook())
    elif CVARHook().can_handle(agent):
        hooks.append(CVARHook())

    # Always add standard PPO hook as fallback
    hooks.append(StandardPPOHook())

    return hooks


def _collect_rollouts(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: Optional[int] = None,
    last_obs: Optional[np.ndarray] = None,
    last_done: Optional[np.ndarray] = None,
    last_scores: Optional[np.ndarray] = None,
    last_info: Optional[Dict[str, Any]] = None,
    *,
    recurrent: bool,
    reset_on_collect: bool = True,
) -> Tuple[List[float], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Collect rollouts for on-policy algorithms using a modular hook system."""
    if not getattr(agent, "use_rollout_buffer", False):
        raise RuntimeError(
            "collect_rollouts can only be used when use_rollout_buffer=True"
        )

    # Initialize timing tracker if not present
    if not hasattr(agent, "timing_tracker"):
        agent.timing_tracker = TimingTracker()

    # Get appropriate hooks for this agent
    hooks = get_rollout_hooks(agent)
    primary_hook = hooks[0]  # First hook handles the main logic

    n_steps = n_steps or agent.learn_step

    # Use timing tracker context manager
    with agent.timing_tracker.time_context("rollout_collection"):
        if reset_on_collect or (
            last_obs is None
            or last_done is None
            or last_scores is None
            or last_info is None
        ):
            # Initial reset
            obs, info = env.reset()
            scores = np.zeros(agent.num_envs)
            done = np.zeros(agent.num_envs)

            # Reset agent hidden state
            agent.hidden_state = (
                agent.get_initial_hidden_state(agent.num_envs) if recurrent else None
            )
        else:
            # Continue from last state
            obs = last_obs
            done = last_done
            scores = last_scores
            info = last_info

        agent.rollout_buffer.reset()
        current_hidden_state_for_actor = agent.hidden_state

        # Initialize step data with hook-specific setup
        step_data = {}
        for hook in hooks:
            hook_data = hook.on_rollout_start(agent, env, n_steps, reset_on_collect)
            step_data.update(hook_data)

        completed_episode_scores = []
        for _ in range(n_steps):
            current_hidden_state_for_buffer = current_hidden_state_for_actor
            step_data["current_hidden_state_for_buffer"] = (
                current_hidden_state_for_buffer
            )

            # Process step start
            for hook in hooks:
                step_data = hook.on_step_start(agent, obs, info, step_data)

            # Get action, statistics and (maybe) recurrent hidden state from agent
            if recurrent:
                action_result = agent.get_action(
                    obs,
                    action_mask=info.get("action_mask", None),
                    hidden_state=current_hidden_state_for_actor,
                )
            else:
                action_result = agent.get_action(
                    obs, action_mask=info.get("action_mask", None)
                )

            # Process action result through hooks
            processed_result = primary_hook.process_action_result(
                agent, action_result, recurrent
            )

            if recurrent:
                if len(processed_result) >= 6:
                    (
                        action,
                        log_prob,
                        _,
                        value,
                        next_hidden_for_actor,
                        encoder_output,
                    ) = processed_result[:6]
                else:
                    action, log_prob, _, value, next_hidden_for_actor = (
                        processed_result[:5]
                    )
                    encoder_output = None
                agent.hidden_state = next_hidden_for_actor
            else:
                if len(processed_result) >= 6:
                    action, log_prob, _, value, _, encoder_output = processed_result[:6]
                else:
                    action, log_prob, _, value = processed_result[:4]
                    encoder_output = None

            step_data["encoder_output"] = encoder_output

            # Clip action to action space
            policy = getattr(agent, agent.registry.policy())
            if isinstance(policy, StochasticActor) and isinstance(
                agent.action_space, spaces.Box
            ):
                if policy.squash_output:
                    clipped_action = policy.scale_action(action)
                else:
                    clipped_action = np.clip(
                        action,
                        agent.action_space.low,
                        agent.action_space.high,
                    )
            else:
                clipped_action = action

            next_obs, reward, term, trunc, next_info = env.step(clipped_action)

            # Process reward through hooks
            processed_reward = primary_hook.process_reward(
                agent, reward, obs, next_obs, action, step_data
            )

            # Check if termination condition is met
            if isinstance(term, (list, np.ndarray)):
                is_terminal = (
                    np.logical_or(term, trunc)
                    if isinstance(trunc, (list, np.ndarray))
                    else term
                )
            else:
                is_terminal = term or trunc

            # Prepare buffer data through primary hook
            buffer_data = primary_hook.prepare_buffer_data(
                agent,
                obs,
                action,
                processed_reward,
                is_terminal,
                value,
                log_prob,
                next_obs,
                current_hidden_state_for_buffer,
                step_data,
            )

            # Add to buffer through primary hook
            primary_hook.add_to_buffer(agent, buffer_data)

            scores += np.atleast_1d(processed_reward)
            done = np.atleast_1d(is_terminal)

            if recurrent and np.any(done):
                finished_mask = done.astype(bool)
                initial_hidden_states_for_reset = agent.get_initial_hidden_state(
                    agent.num_envs
                )
                if isinstance(agent.hidden_state, dict):
                    for key in agent.hidden_state:
                        reset_states_for_key = initial_hidden_states_for_reset[key][
                            :, finished_mask, :
                        ]
                        if reset_states_for_key.shape[1] > 0:
                            agent.hidden_state[key][
                                :, finished_mask, :
                            ] = reset_states_for_key

            # Handle episode endings through hooks
            for hook in hooks:
                hook.on_episode_end(agent, 0, done, step_data)

            if recurrent:
                current_hidden_state_for_actor = agent.hidden_state

            obs = next_obs
            info = next_info

            for idx, env_done in enumerate(done):
                if env_done:
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0

        # Handle rollout end through hooks
        for hook in hooks:
            hook.on_rollout_end(agent, env, step_data)

        # Store the last observation and info for potential continuation
        agent._last_obs = (obs, info)

        # Calculate last value to compute returns and advantages properly
        with torch.no_grad():
            if recurrent:
                action_result = agent._get_action_and_values(
                    agent.preprocess_observation(obs),
                    hidden_state=agent.hidden_state,
                )
            else:
                action_result = agent._get_action_and_values(
                    agent.preprocess_observation(obs)
                )
            last_value = action_result[3]

            last_value = last_value.cpu().numpy()
            last_done = np.atleast_1d(term)

        agent.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value, last_done=last_done
        )

    # Update timing metrics using the timing tracker
    agent.last_collection_time = agent.timing_tracker.get_average_time(
        "rollout_collection"
    )
    agent.total_collection_time = agent.timing_tracker.get_time("rollout_collection")

    return completed_episode_scores, obs, done, scores, info


def collect_rollouts(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: Optional[int] = None,
    reset_on_collect: bool = True,
    **kwargs,
) -> List[float]:
    """Collect rollouts for non-recurrent on-policy algorithms using modular hooks.

    This function automatically detects the agent type and applies appropriate
    collection logic through a hook system:
    - ICM agents: Intrinsic reward calculation and encoder state management
    - CVAR agents: CVaR update terms and episode return tracking
    - Standard PPO: Basic rollout collection

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicy
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: Optional[int]
    :param reset_on_collect: Whether to reset the environment and agent state before collecting. Defaults to True.
    :type reset_on_collect: bool

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: List[float]
    """
    # Use stored state if not resetting and available
    last_obs, last_info = (
        agent._last_obs
        if hasattr(agent, "_last_obs") and agent._last_obs
        else (None, None)
    )

    completed_scores, _, _, _, _ = _collect_rollouts(
        agent,
        env,
        n_steps,
        last_obs=last_obs,
        last_done=None,
        last_scores=None,
        last_info=last_info,
        recurrent=False,
        reset_on_collect=reset_on_collect,
        **kwargs,
    )
    return completed_scores


def collect_rollouts_recurrent(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: Optional[int] = None,
    reset_on_collect: bool = True,
    **kwargs,
) -> List[float]:
    """Collect rollouts for recurrent on-policy algorithms using modular hooks.

    This function automatically detects the agent type and applies appropriate
    collection logic through a hook system:
    - ICM agents: Intrinsic reward calculation and encoder state management
    - CVAR agents: CVaR update terms and episode return tracking
    - Standard PPO: Basic rollout collection

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicy
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: Optional[int]
    :param reset_on_collect: Whether to reset the environment and agent state before collecting. Defaults to True.
    :type reset_on_collect: bool

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: List[float]
    """
    # Use stored state if not resetting and available
    last_obs, last_info = (
        agent._last_obs
        if hasattr(agent, "_last_obs") and agent._last_obs
        else (None, None)
    )

    completed_scores, _, _, _, _ = _collect_rollouts(
        agent,
        env,
        n_steps,
        last_obs=last_obs,
        last_done=None,
        last_scores=None,
        last_info=last_info,
        recurrent=True,
        reset_on_collect=reset_on_collect,
        **kwargs,
    )
    return completed_scores
