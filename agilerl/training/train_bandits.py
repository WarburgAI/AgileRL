import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import wandb
from accelerate import Accelerator
from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.components.data import ReplayDataset
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    default_progress_bar,
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = Optional[Dict[str, Any]]
PopulationType = List[RLAlgorithm]


def train_bandits(
    env: gym.Env,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: ReplayBuffer,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 20000,
    episode_steps: int = 500,
    evo_steps: int = 2500,
    eval_steps: int = 500,
    eval_loop: int = 1,
    target: Optional[float] = None,
    tournament: Optional[TournamentSelection] = None,
    mutation: Optional[Mutations] = None,
    checkpoint: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path: Optional[str] = None,
    wb: bool = False,
    verbose: bool = True,
    accelerator: Optional[Accelerator] = None,
    wandb_api_key: Optional[str] = None,
) -> Tuple[PopulationType, List[List[float]]]:
    """The general bandit training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param INIT_HP: Dictionary containing initial hyperparameters, defaults to None
    :type INIT_HP: dict, optional
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param max_steps: Maximum number of steps in environment, defaults to 20000
    :type max_steps: int, optional
    :param episode_steps: Number of steps in environment per episode, defaults to 500
    :type episode_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 2500
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode, defaults to 500
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (steps), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param overwrite_checkpoints: Overwrite previous checkpoints during training, defaults to False
    :type overwrite_checkpoints: bool, optional
    :param save_elite: Boolean flag indicating whether to save elite member at the end
        of training, defaults to False
    :type save_elite: bool, optional
    :param elite_path: Location to save elite agent, defaults to None
    :type elite_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional
    """
    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb, bool
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
        )

    if wb:
        init_wandb(
            algo=algo,
            env_name=env_name,
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
            accelerator=accelerator,
            project="AgileRL-Bandits",
        )

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(dataset=replay_dataset, dataloader=replay_dataloader)
    else:
        sampler = Sampler(memory=memory)

    # Pre-training mutation
    if accelerator is None:
        if mutation is not None:
            pop = mutation.mutation(pop, pre_training_mut=True)

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    # Format progress bar
    pbar = default_progress_bar(max_steps, accelerator)

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    checkpoint_count = 0
    evo_count = 0

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()
        pop_episode_scores = []
        pop_fps = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            score = 0
            losses = []
            context = env.reset()  # Reset environment at start of episode

            start_time = time.time()
            for idx_step in range(episode_steps):
                if swap_channels:
                    context = obs_channels_to_first(context)
                # Get next action from agent
                action = agent.get_action(context)
                next_context, reward = env.step(action)  # Act in environment

                # Save experience to replay buffer
                transition = TensorDict(
                    {
                        "obs": context,
                        "reward": reward,
                    }
                )
                transition = transition.unsqueeze(0)
                transition.batch_size = [1]
                memory.add(transition)

                # Learn according to learning frequency
                if len(memory) >= agent.batch_size:
                    for _ in range(agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        experiences = sampler.sample(agent.batch_size)
                        loss = agent.learn(experiences)
                        losses.append(loss)

                score += reward
                agent.regret.append(agent.regret[-1] + 1 - reward)

                context = next_context

            agent.scores.append(score)
            pop_episode_scores.append(score)
            pop_loss[agent_idx].append(np.mean(losses))
            agent.steps[-1] += episode_steps
            fps = episode_steps / (time.time() - start_time)
            pop_fps.append(fps)
            total_steps += episode_steps
            pbar.update(episode_steps // len(pop))

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = np.mean(pop_episode_scores)
        regrets = [agent.regret[-1] for agent in pop]
        mean_losses = np.mean([losses[-10:] for losses in pop_loss], axis=1)

        if wb:
            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
                "steps_per_agent": total_steps / len(pop),
                "train/mean_score": np.mean(mean_scores),
                "train/mean_regret": np.mean(regrets),
                "train/best_regret": np.min(regrets),
                "train/mean_loss": np.mean(mean_losses),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }
            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb.log(wandb_dict)
                accelerator.wait_for_everyone()
            else:
                wandb.log(wandb_dict)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # Early stop if consistently reaches target
        if target is not None:
            if (
                np.all(
                    np.greater([np.mean(agent.fitness[-10:]) for agent in pop], target)
                )
                and len(pop[0].steps) >= 100
            ):
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            if pop[0].steps[-1] // evo_steps > evo_count:
                pop = tournament_selection_and_mutation(
                    population=pop,
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env_name,
                    algo=algo,
                    elite_path=elite_path,
                    save_elite=save_elite,
                    accelerator=accelerator,
                )
                evo_count += 1

        if verbose:
            regret = ["%.2f" % regret for regret in regrets]
            avg_regret = "%.2f" % np.mean(np.array(regrets))
            fitness = ["%.2f" % fitness for fitness in fitnesses]
            avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
            avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]

            banner_text = f"Global Steps {total_steps}"
            banner_width = max(len(banner_text) + 8, 35)
            border = "=" * banner_width
            centered_text = f"{banner_text}".center(banner_width)
            pbar.write(
                f"{border}\n"
                f"{centered_text}\n"
                f"{border}\n"
                f"Regret:\t\t{regret}\n"
                f"Mean regret:\t{avg_regret}\n"
                f"Fitness:\t\t{fitness}\n"
                f"5 fitness avgs:\t{avg_fitness}\n"
                f"10 score avgs:\t{avg_score}\n"
                f"Agents:\t\t{agents}\n"
                f"Steps:\t\t{num_steps}\n"
                f"Mutations:\t\t{muts}"
            )

        # Save model checkpoint
        if checkpoint is not None:
            if pop[0].steps[-1] // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=pop,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    pbar.close()
    return pop, pop_fitnesses
