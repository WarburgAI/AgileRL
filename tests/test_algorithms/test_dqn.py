import copy
from pathlib import Path

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.dqn import DQN
from agilerl.components.data import Transition
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
    get_experiences_batch,
    get_sample_from_space,
)


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


class DummyDQN(DQN):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)

        self.tensor_test = torch.randn(1)


class DummyEnv:
    def __init__(self, observation_space, vect=True, num_envs=2):
        self.observation_space = observation_space.shape
        self.vect = vect
        if self.vect:
            self.observation_space = (num_envs,) + self.observation_space
            self.n_envs = num_envs
            self.num_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.observation_space), {}

    def step(self, action):
        return (
            np.random.rand(*self.observation_space),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


@pytest.fixture
def simple_mlp():
    network = nn.Sequential(
        nn.Linear(4, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Tanh(),
    )
    return network


@pytest.fixture
def simple_cnn():
    network = nn.Sequential(
        nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 3 (for RGB images), Output channels: 16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 16, Output channels: 32
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),  # Flatten the 2D feature map to a 1D vector
        nn.Linear(32 * 16 * 16, 128),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(128, 1),  # Output layer with num_classes output features
    )
    return network


# initialize DQN with valid parameters
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,)), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=255), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 2, dict_space=True), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 2, dict_space=False), EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_dqn(observation_space, encoder_cls, accelerator):
    action_space = generate_discrete_space(2)
    dqn = DQN(observation_space, action_space, accelerator=accelerator)

    expected_device = accelerator.device if accelerator else "cpu"
    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == expected_device
    assert dqn.accelerator == accelerator
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.double is False
    # assert dqn.actor_network is None
    assert isinstance(dqn.actor.encoder, encoder_cls)
    assert isinstance(dqn.actor_target.encoder, encoder_cls)
    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(dqn.optimizer.optimizer, expected_opt_cls)
    assert isinstance(dqn.criterion, nn.MSELoss)


# Can initialize DQN with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (generate_random_box_space(shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (
            generate_random_box_space(shape=(3, 64, 64), low=0, high=255),
            "simple_cnn",
            torch.randn(1, 3, 64, 64),
        ),
    ],
)
def test_initialize_dqn_with_actor_network_make_evo(
    observation_space, actor_network, input_tensor, request
):
    action_space = generate_discrete_space(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    dqn = DQN(observation_space, action_space, actor_network=actor_network)

    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == "cpu"
    assert dqn.accelerator is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.double is False
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert isinstance(dqn.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_space, net_type",
    [
        (generate_random_box_space(shape=(4,)), "mlp"),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=255), "cnn"),
    ],
)
def test_initialize_dqn_with_actor_network_evo_net(observation_space, net_type):
    action_space = generate_discrete_space(2)
    if net_type == "mlp":
        actor_network = EvolvableMLP(
            num_inputs=observation_space.shape[0],
            num_outputs=action_space.n,
            hidden_size=[64, 64],
            activation="ReLU",
        )
    else:
        actor_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            activation="ReLU",
        )

    dqn = DQN(observation_space, action_space, actor_network=actor_network)

    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == "cpu"
    assert dqn.accelerator is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.double is False
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert isinstance(dqn.criterion, nn.MSELoss)


def test_initialize_dqn_with_incorrect_actor_net_type():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    actor_network = "dummy"

    with pytest.raises(TypeError) as a:
        dqn = DQN(observation_space, action_space, actor_network=actor_network)

        assert dqn
        assert (
            str(a.value)
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type nn.Module."
        )


# Returns the expected action when given a state observation and epsilon=0 or 1.
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_discrete_space(4),
        generate_random_box_space(shape=(4,)),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        generate_multidiscrete_space(2, 2),
        generate_dict_or_tuple_space(2, 2, dict_space=True),
        generate_dict_or_tuple_space(2, 2, dict_space=False),
    ],
)
def test_returns_expected_action_epsilon_greedy(observation_space):
    action_space = generate_discrete_space(2)

    dqn = DQN(observation_space, action_space)
    state = get_sample_from_space(observation_space)

    action_mask = None

    epsilon = 0
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n

    epsilon = 1
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n


# Returns the expected action when given a state observation and action mask.
def test_returns_expected_action_mask():
    accelerator = Accelerator()
    observation_space = spaces.Discrete(4)
    action_space = generate_discrete_space(2)

    dqn = DQN(observation_space, action_space, accelerator=accelerator)
    state = get_sample_from_space(observation_space)

    action_mask = np.array([0, 1])

    epsilon = 0
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1

    epsilon = 1
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1


def test_returns_expected_action_mask_vectorized():
    accelerator = Accelerator()
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = DQN(observation_space, action_space, accelerator=accelerator)
    state = get_sample_from_space(observation_space, batch_size=2)

    action_mask = np.array([[0, 1], [1, 0]])

    epsilon = 0
    action = dqn.get_action(state, epsilon, action_mask)

    assert np.array_equal(action, [1, 0])

    epsilon = 1
    action = dqn.get_action(state, epsilon, action_mask)

    assert np.array_equal(action, [1, 0])


def test_dqn_optimizer_parameters():
    observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(2)
    dqn = DQN(observation_space, action_space)

    # Store initial parameters
    initial_params = {
        name: param.clone() for name, param in dqn.actor.named_parameters()
    }

    # Perform a dummy optimization step
    dummy_input = torch.randn(1, 4)
    dummy_return = torch.tensor([1.0])

    q_eval = dqn.actor(dummy_input)
    loss = (dummy_return - q_eval) ** 2
    loss = loss.mean()
    dqn.optimizer.zero_grad()
    loss.backward()
    dqn.optimizer.step()

    # Check if parameters have changed
    not_updated = []
    for name, param in dqn.actor.named_parameters():
        if torch.equal(initial_params[name], param):
            not_updated.append(name)

    assert not not_updated, f"The following parameters weren't updated:\n{not_updated}"


# learns from experiences and updates network parameters
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_discrete_space(4),
        generate_random_box_space(shape=(4,)),
        generate_multidiscrete_space(2, 2),
        generate_dict_or_tuple_space(2, 2, dict_space=True),
        generate_dict_or_tuple_space(2, 2, dict_space=False),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("double", [False, True])
def test_learns_from_experiences(observation_space, accelerator, double):
    action_space = generate_discrete_space(2)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = DQN(
        observation_space,
        action_space,
        batch_size=batch_size,
        accelerator=accelerator,
        double=double,
    )

    # Create a batch of experiences
    device = accelerator.device if accelerator else "cpu"
    experiences = get_experiences_batch(
        observation_space, action_space, batch_size, device
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))

    # Call the learn method
    loss = dqn.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    net_config = {"encoder_config": {"hidden_size": [64, 64]}}
    batch_size = 64
    lr = 1e-4
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    double = False
    actor_network = None
    device = "cpu"
    accelerator = None
    wrap = True

    dqn = DQN(
        observation_space,
        action_space,
        net_config=net_config,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        double=double,
        actor_network=actor_network,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
    )

    dqn.soft_update()

    eval_params = list(dqn.actor.parameters())
    target_params = list(dqn.actor_target.parameters())
    expected_params = [
        dqn.tau * eval_param + (1.0 - dqn.tau) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )


# Runs algorithm test loop
def test_algorithm_test_loop():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    num_envs = 3

    env = DummyEnv(observation_space=observation_space, vect=True, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = DQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    env = DummyEnv(observation_space=observation_space, vect=False)

    agent = DQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=255)
    action_space = generate_discrete_space(2)

    env = DummyEnv(observation_space=observation_space, vect=True)

    net_config_cnn = {
        "encoder_config": {"channel_size": [3], "kernel_size": [3], "stride_size": [1]}
    }

    agent = DQN(
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorized images
def test_algorithm_test_loop_images_unvectorized():
    observation_space = spaces.Box(0, 1, shape=(32, 32, 3))
    action_space = generate_discrete_space(2)

    env = DummyEnv(observation_space=observation_space, vect=False)

    net_config_cnn = {
        "encoder_config": {"channel_size": [3], "kernel_size": [3], "stride_size": [1]}
    }

    agent = DQN(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = DummyDQN(observation_space, action_space)
    dqn.tensor_attribute = torch.randn(1)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    # assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores
    assert clone_agent.tensor_attribute == dqn.tensor_attribute
    assert clone_agent.tensor_test == dqn.tensor_test

    accelerator = Accelerator()
    dqn = DQN(observation_space, action_space, accelerator=accelerator)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    # assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores

    accelerator = Accelerator()
    dqn = DQN(observation_space, action_space, accelerator=accelerator, wrap=False)
    clone_agent = dqn.clone(wrap=False)

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    # assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())

    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores


def test_clone_new_index():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = DummyDQN(observation_space, action_space)
    clone_agent = dqn.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    batch_size = 8
    dqn = DQN(observation_space, action_space)

    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randint(0, 2, (batch_size, 1))
    rewards = torch.rand(batch_size, 1)
    next_states = torch.randn(batch_size, observation_space.shape[0])
    dones = torch.zeros(batch_size, 1)

    experiences = Transition(
        obs=states,
        action=actions,
        reward=rewards,
        next_obs=next_states,
        done=dones,
    ).to_tensordict()
    dqn.learn(experiences)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())

    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    dqn = DQN(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=generate_discrete_space(2),
        accelerator=Accelerator(),
    )
    dqn.unwrap_models()
    assert isinstance(dqn.actor, nn.Module)
    assert isinstance(dqn.actor_target, nn.Module)


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,)), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=255), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 2, dict_space=True), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 2, dict_space=False), EvolvableMultiInput),
    ],
)
def test_save_load_checkpoint_correct_data_and_format(
    observation_space, encoder_cls, tmpdir
):
    # Initialize the DQN agent
    dqn = DQN(
        observation_space=observation_space,
        action_space=generate_discrete_space(2),
    )

    initial_actor_state_dict = dqn.actor.state_dict()
    init_optim_state_dict = dqn.optimizer.state_dict()

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill, weights_only=False)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    dqn = DQN(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=generate_discrete_space(2),
    )
    # Load checkpoint
    dqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(dqn.actor.encoder, encoder_cls)
    assert isinstance(dqn.actor_target.encoder, encoder_cls)
    assert dqn.lr == 1e-4
    # assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(initial_actor_state_dict) == str(dqn.actor.state_dict())
    assert str(init_optim_state_dict) == str(dqn.optimizer.state_dict())
    assert dqn.batch_size == 64
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 1e-3
    assert dqn.mut is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]


# The saved checkpoint file contains the correct data and format.
# TODO: This will be deprecated in the future.
@pytest.mark.parametrize(
    "actor_network, input_tensor",
    [
        ("simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_save_load_checkpoint_correct_data_and_format_cnn_network(
    actor_network, input_tensor, request, tmpdir
):
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the DQN agent
    dqn = DQN(
        observation_space=generate_random_box_space(shape=(3, 64, 64), low=0, high=255),
        action_space=generate_discrete_space(2),
        actor_network=actor_network,
    )

    initial_actor_state_dict = dqn.actor.state_dict()
    init_optim_state_dict = dqn.optimizer.state_dict()

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill, weights_only=False)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    dqn = DQN(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=generate_discrete_space(2),
    )
    # Load checkpoint
    dqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(dqn.actor, nn.Module)
    assert isinstance(dqn.actor_target, nn.Module)
    assert dqn.lr == 1e-4
    # assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(initial_actor_state_dict) == str(dqn.actor.state_dict())
    assert str(init_optim_state_dict) == str(dqn.optimizer.state_dict())
    assert dqn.batch_size == 64
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 1e-3
    assert dqn.mut is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,)), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=255), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 2, dict_space=True), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 2, dict_space=False), EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_load_from_pretrained(
    observation_space, encoder_cls, device, accelerator, tmpdir
):
    # Initialize the DQN agent
    dqn = DQN(
        observation_space=observation_space,
        action_space=generate_discrete_space(2),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_dqn = DQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_dqn.observation_space == dqn.observation_space
    assert new_dqn.action_space == dqn.action_space
    assert isinstance(new_dqn.actor.encoder, encoder_cls)
    assert isinstance(new_dqn.actor_target.encoder, encoder_cls)
    assert new_dqn.lr == dqn.lr
    assert str(new_dqn.actor.to("cpu").state_dict()) == str(dqn.actor.state_dict())
    assert new_dqn.batch_size == dqn.batch_size
    assert new_dqn.learn_step == dqn.learn_step
    assert new_dqn.gamma == dqn.gamma
    assert new_dqn.tau == dqn.tau
    assert new_dqn.mut == dqn.mut
    assert new_dqn.index == dqn.index
    assert new_dqn.scores == dqn.scores
    assert new_dqn.fitness == dqn.fitness
    assert new_dqn.steps == dqn.steps


# The saved checkpoint file contains the correct data and format.
# TODO: This will be deprecated in the future.
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (generate_random_box_space(shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (
            generate_random_box_space(shape=(3, 64, 64), low=0, high=255),
            "simple_cnn",
            torch.randn(1, 3, 64, 64),
        ),
    ],
)
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = generate_discrete_space(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the DQN agent
    dqn = DQN(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_dqn = DQN.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_dqn.observation_space == dqn.observation_space
    assert new_dqn.action_space == dqn.action_space
    assert isinstance(new_dqn.actor, nn.Module)
    assert isinstance(new_dqn.actor_target, nn.Module)
    assert new_dqn.lr == dqn.lr
    assert str(new_dqn.actor.to("cpu").state_dict()) == str(dqn.actor.state_dict())
    assert new_dqn.batch_size == dqn.batch_size
    assert new_dqn.learn_step == dqn.learn_step
    assert new_dqn.gamma == dqn.gamma
    assert new_dqn.tau == dqn.tau
    assert new_dqn.mut == dqn.mut
    assert new_dqn.index == dqn.index
    assert new_dqn.scores == dqn.scores
    assert new_dqn.fitness == dqn.fitness
    assert new_dqn.steps == dqn.steps
