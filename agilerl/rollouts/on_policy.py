"""Functions for collecting rollouts for on-policy algorithms."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from agilerl.algorithms import PPO
from agilerl.networks import StochasticActor
from agilerl.typing import GymEnvType
from agilerl.utils.metrics import TimingTracker

SupportedOnPolicy = PPO


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
    """Collect rollouts for on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicy
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for. Defaults to agent.learn_step if not provided.
    :type n_steps: Optional[int]
    :param last_obs: The observation to use for the first step. Defaults to None, where the environment is reset.
    :type last_obs: Optional[np.ndarray]
    :param last_done: The done flag to use for the first step. Defaults to None, where the environment is reset.
    :type last_done: Optional[np.ndarray]
    :param last_scores: The scores to use for the first step. Defaults to None, where the environment is reset.
    :type last_scores: Optional[np.ndarray]
    :param last_info: The info for the current step. Defaults to None, where the environment is reset.
    :type last_info: Optional[Dict[str, Any]]
    :param recurrent: Whether the agent is recurrent.
    :type recurrent: bool
    :param reset_on_collect: Whether to reset the environment and agent state before collecting. Defaults to True.
    :type reset_on_collect: bool

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: List[float]
    :return: The observation, done flag, scores, and info for the current step.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]
    """
    if not getattr(agent, "use_rollout_buffer", False):
        raise RuntimeError(
            "collect_rollouts can only be used when use_rollout_buffer=True"
        )

    # Initialize timing tracker if not present
    if not hasattr(agent, "_timing_tracker"):
        agent._timing_tracker = TimingTracker()

    n_steps = n_steps or agent.learn_step

    # Use timing tracker context manager
    with agent._timing_tracker.time_context("rollout_collection"):
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

        completed_episode_scores = []
        for _ in range(n_steps):
            current_hidden_state_for_buffer = current_hidden_state_for_actor

            # Get action, statistics and (maybe) recurrent hidden state from agent
            if recurrent:
                action, log_prob, _, value, next_hidden_for_actor = agent.get_action(
                    obs,
                    action_mask=info.get("action_mask", None),
                    hidden_state=current_hidden_state_for_actor,
                )
                agent.hidden_state = next_hidden_for_actor
            else:
                action, log_prob, _, value = agent.get_action(
                    obs, action_mask=info.get("action_mask", None)
                )

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

            # Check if termination condition is met
            if isinstance(term, (list, np.ndarray)):
                is_terminal = (
                    np.logical_or(term, trunc)
                    if isinstance(trunc, (list, np.ndarray))
                    else term
                )
            else:
                is_terminal = term or trunc

            reward_np = np.atleast_1d(reward)
            is_terminal_np = np.atleast_1d(is_terminal)
            value_np = np.atleast_1d(value)
            log_prob_np = np.atleast_1d(log_prob)

            agent.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward_np,
                done=done,
                value=value_np,
                log_prob=log_prob_np,
                next_obs=next_obs,
                hidden_state=current_hidden_state_for_buffer,
                episode_start=is_terminal_np,
            )

            scores += reward_np
            done = is_terminal_np

            if recurrent and np.any(is_terminal_np):
                finished_mask = is_terminal_np.astype(bool)
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

            if recurrent:
                current_hidden_state_for_actor = agent.hidden_state

            obs = next_obs
            info = next_info

            for idx, env_done in enumerate(is_terminal_np):
                if env_done:
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0

        # Store the last observation and info for potential continuation
        agent._last_obs = (obs, info)

        # Calculate last value to compute returns and advantages properly
        # TODO: We shouldn't access a hidden method here...
        with torch.no_grad():
            if recurrent:
                _, _, _, last_value, _ = agent._get_action_and_values(
                    agent.preprocess_observation(obs),
                    hidden_state=agent.hidden_state,
                )
            else:
                _, _, _, last_value, _ = agent._get_action_and_values(
                    agent.preprocess_observation(obs)
                )

            last_value = last_value.cpu().numpy()
            last_done = np.atleast_1d(term)

        agent.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value, last_done=last_done
        )

    # Update timing metrics using the timing tracker
    agent.last_collection_time = agent._timing_tracker.get_average_time(
        "rollout_collection"
    )
    agent.total_collection_time = agent._timing_tracker.get_time("rollout_collection")

    return completed_episode_scores, obs, done, scores, info


def collect_rollouts(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: Optional[int] = None,
    reset_on_collect: bool = True,
    **kwargs,
) -> List[float]:
    """Collect rollouts for non-recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
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
    """Collect rollouts for recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
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
