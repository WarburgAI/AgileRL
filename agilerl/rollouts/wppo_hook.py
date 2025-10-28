"""WPPO-specific rollout collection hook."""

from typing import Any, Dict

import numpy as np

from .on_policy import RolloutHook


class WPPOHook(RolloutHook):
    """
    Hook for WPPO-specific rollout collection logic.

    Handles:
    - Extraction of reward components from environment infos
    - Extraction of dual multipliers from environment infos
    - Adding these to the rollout buffer
    - Calling decomposed advantage computation instead of standard GAE
    """

    def can_handle(self, agent) -> bool:
        """Check if this hook can handle the given agent."""
        # WPPO has _compute_decomposed_advantages method AND WPPORolloutBuffer
        # If WPPO is using standard RolloutBuffer (for testing), don't use this hook
        from warburgai.agents.algorithms.wppo_rollout_buffer import WPPORolloutBuffer

        return (
            hasattr(agent, "_compute_decomposed_advantages")
            and hasattr(agent, "rollout_buffer")
            and isinstance(agent.rollout_buffer, WPPORolloutBuffer)
        )

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
        timeout=None,
    ) -> Dict[str, Any]:
        """Prepare data for buffer addition, including reward components and dual multipliers."""
        # Get standard buffer data
        buffer_data = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "value": value,
            "log_prob": log_prob,
            "next_obs": next_obs,
            "hidden_state": hidden_state,
            "timeouts": timeout,
        }

        # Extract reward components if available in step_data (from info)
        if "reward_components" in step_data:
            buffer_data["reward_components"] = step_data["reward_components"]

        # Extract dual multipliers if available
        if "dual_multipliers" in step_data:
            buffer_data["dual_multipliers"] = step_data["dual_multipliers"]

        # Extract value components if agent computed them
        if "value_components" in step_data:
            buffer_data["value_components"] = step_data["value_components"]

        # CRITICAL FIX: Include all other step_data (like action_mask) to avoid KL divergence
        # Action masks are added to step_data and must be passed through to the buffer
        if "action_mask" in step_data:
            buffer_data["action_mask"] = step_data["action_mask"]

        return buffer_data

    def add_to_buffer(self, agent, buffer_data: Dict[str, Any]) -> None:
        """Add data to the WPPO rollout buffer with reward components."""
        # Ensure arrays are properly shaped
        reward_np = np.atleast_1d(buffer_data["reward"])
        done_np = np.atleast_1d(buffer_data["done"])
        value_np = np.atleast_1d(buffer_data["value"])
        log_prob_np = np.atleast_1d(buffer_data["log_prob"])

        # Extract WPPO-specific data
        reward_components = buffer_data.get("reward_components", None)
        dual_multipliers = buffer_data.get("dual_multipliers", None)
        value_components = buffer_data.get("value_components", None)

        # Add to buffer (WPPORolloutBuffer has extended add() method)
        agent.rollout_buffer.add(
            obs=buffer_data["obs"],
            action=buffer_data["action"],
            reward=reward_np,
            done=done_np,
            value=value_np,
            log_prob=log_prob_np,
            next_obs=buffer_data["next_obs"],
            hidden_state=buffer_data["hidden_state"],
            episode_start=np.atleast_1d(buffer_data["episode_start"]),
            action_mask=buffer_data.get("action_mask", None),
            timeouts=buffer_data.get("timeouts", np.zeros(agent.num_envs, dtype=bool)),
            # WPPO-specific parameters
            reward_components=reward_components,
            dual_multipliers=dual_multipliers,
            value_components=value_components,
        )

    def on_step_start(
        self, agent, obs, info, step_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract reward components and dual multipliers from info at step start."""
        # Extract reward components if environment provides them
        if isinstance(info, dict) and "reward_components" in info:
            step_data["reward_components"] = info["reward_components"]

        # Extract dual multipliers
        if isinstance(info, dict) and "dual_multipliers" in info:
            step_data["dual_multipliers"] = info["dual_multipliers"]

        # Compute value components if agent is WPPO
        # This happens during action selection, so we'll store them
        # Note: Value components are computed during get_action, we just need to track them

        return step_data
