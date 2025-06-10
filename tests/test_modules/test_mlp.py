import copy

import numpy as np
import pytest
import torch

from agilerl.modules.custom_components import NoisyLinear
from agilerl.modules.mlp import EvolvableMLP


######### Define fixtures #########
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


def test_noisy_linear(device):
    noisy_linear = NoisyLinear(2, 10).to(device)
    noisy_linear.training = False
    with torch.no_grad():
        output = noisy_linear.forward(torch.randn(1, 2).to(device))
    assert output.shape == (1, 10)


######### Test instantiation #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(10, 5, [32, 64, 128]), (2, 1, [32]), (100, 3, [8, 8, 8, 8, 8, 8, 8])],
)
def test_instantiation(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    assert isinstance(evolvable_mlp, EvolvableMLP)


@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(0, 20, [16]), (20, 0, [16]), (10, 2, []), (10, 2, [0])],
)
def test_incorrect_instantiation(num_inputs, num_outputs, hidden_size, device):
    with pytest.raises(Exception):
        EvolvableMLP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_size=hidden_size,
            device=device,
        )


@pytest.mark.parametrize(
    "activation, output_activation",
    [
        ("ELU", "Softmax"),
        ("Tanh", "PReLU"),
        ("LeakyReLU", "GELU"),
        ("ReLU", "Sigmoid"),
        ("Tanh", "Softplus"),
        ("Tanh", "Softsign"),
    ],
)
def test_instantiation_with_different_activations(
    activation, output_activation, device
):
    evolvable_mlp = EvolvableMLP(
        num_inputs=6,
        num_outputs=4,
        hidden_size=[32],
        activation=activation,
        output_activation=output_activation,
        output_vanish=True,
        device=device,
    )
    assert isinstance(evolvable_mlp, EvolvableMLP)


def test_reset_noise(device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=10,
        num_outputs=4,
        hidden_size=[32, 32],
        output_vanish=True,
        noisy=True,
        device=device,
    )
    evolvable_mlp.reset_noise()
    assert isinstance(evolvable_mlp.model[0], NoisyLinear)


######### Test forward #########
@pytest.mark.parametrize(
    "input_tensor, num_inputs, num_outputs, hidden_size, output_size",
    [
        (torch.randn(1, 10), 10, 5, [32, 64, 128], (1, 5)),
        (torch.randn(1, 2), 2, 1, [32], (1, 1)),
        (torch.randn(1, 100), 100, 3, [8, 8, 8, 8, 8, 8, 8], (1, 3)),
        (np.random.randn(1, 100), 100, 3, [8, 8, 8, 8, 8, 8, 8], (1, 3)),
    ],
)
def test_forward(
    input_tensor, num_inputs, num_outputs, hidden_size, output_size, device
):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = evolvable_mlp.forward(input_tensor)
    assert output_tensor.shape == output_size


######### Test add_mlp_layer #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [
        (10, 5, [32, 64, 128]),
        (2, 1, [32]),
        (100, 3, [8, 8, 8, 8, 8, 8, 8]),
        (10, 4, [16] * 10),
    ],
)
def test_add_layer(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        max_hidden_layers=10,
        device=device,
    )

    initial_hidden_size = len(evolvable_mlp.hidden_size)
    initial_net = evolvable_mlp.model
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_mlp.add_layer()
    new_net = evolvable_mlp.model
    if initial_hidden_size < 10:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size + 1
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys():
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size


######### Test remove_mlp_layer #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(10, 5, [32, 64, 128]), (2, 1, [32]), (100, 3, [8, 8, 8, 8, 8, 8, 8])],
)
def test_remove_layer(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        min_hidden_layers=1,
        max_hidden_layers=10,
        device=device,
    )

    initial_hidden_size = len(evolvable_mlp.hidden_size)
    initial_net = evolvable_mlp.model
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_mlp.remove_layer()
    new_net = evolvable_mlp.model
    if initial_hidden_size > 1:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size - 1
        for key, param in new_net.named_parameters():
            if (
                key in initial_net_dict.keys()
                and param.shape == initial_net_dict[key].shape
            ):
                torch.testing.assert_close(param, initial_net_dict[key]), evolvable_mlp
    else:
        assert len(evolvable_mlp.hidden_size) == initial_hidden_size


######### Test add_mlp_node #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes",
    [
        (10, 5, [32, 64, 128], None, 4),
        (2, 1, [32], None, None),
        (100, 3, [8, 8, 8, 8, 8, 8, 8], 1, None),
    ],
)
def test_add_nodes(
    num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes, device
):
    mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    original_hidden_size = copy.deepcopy(mlp.hidden_size)
    result = mlp.add_node(hidden_layer=hidden_layer, numb_new_nodes=numb_new_nodes)
    hidden_layer = result["hidden_layer"]
    numb_new_nodes = result["numb_new_nodes"]
    assert (
        mlp.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] + numb_new_nodes
    )


######### Test remove_mlp_node #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes",
    [
        (10, 5, [256, 256, 256], 1, None),
        (2, 1, [32], None, 4),
        (100, 3, [8, 8, 8, 8, 8, 8, 8], None, 4),
    ],
)
def test_remove_nodes(
    num_inputs, num_outputs, hidden_size, hidden_layer, numb_new_nodes, device
):
    mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        min_mlp_nodes=2,
        device=device,
    )
    original_hidden_size = copy.deepcopy(mlp.hidden_size)
    result = mlp.remove_node(numb_new_nodes=numb_new_nodes, hidden_layer=hidden_layer)
    hidden_layer = result["hidden_layer"]
    if numb_new_nodes is None:
        numb_new_nodes = result["numb_new_nodes"]
    assert (
        mlp.hidden_size[hidden_layer]
        == original_hidden_size[hidden_layer] - numb_new_nodes
    )


######### Test clone #########
@pytest.mark.parametrize(
    "num_inputs, num_outputs, hidden_size",
    [(10, 5, [32, 64, 128]), (2, 1, [32]), (100, 3, [8, 8, 8, 8, 8, 8, 8])],
)
def test_clone_instance(num_inputs, num_outputs, hidden_size, device):
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        device=device,
    )
    original_net_dict = dict(evolvable_mlp.model.named_parameters())
    clone = evolvable_mlp.clone()
    clone_net = clone.model
    assert isinstance(clone, EvolvableMLP)
    assert clone.init_dict == evolvable_mlp.init_dict
    assert str(clone.state_dict()) == str(evolvable_mlp.state_dict())
    for key, param in clone_net.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key]), evolvable_mlp


# ---- Batch Norm Tests ----
def check_module_order(model_sequential, target_sequence):
    """
    Helper function to check if a sequence of module types exists in a model.
    target_sequence is a list of tuples, e.g., [(nn.BatchNorm1d, nn.ReLU), (nn.Linear, nn.BatchNorm1d)]
    This means a BatchNorm1d should be followed by a ReLU, and somewhere else a Linear by a BatchNorm1d.
    """
    module_list = list(model_sequential)
    found_all_sequences = True
    for seq in target_sequence:
        found_this_sequence = False
        for i in range(len(module_list) - len(seq) + 1):
            match = True
            for j in range(len(seq)):
                if not isinstance(module_list[i + j], seq[j]):
                    match = False
                    break
            if match:
                found_this_sequence = True
                break
        if not found_this_sequence:
            found_all_sequences = False
            break
    return found_all_sequences

def count_modules_by_type(model_sequential, module_type):
    """Counts the number of modules of a specific type in a sequential model."""
    count = 0
    for module in model_sequential:
        if isinstance(module, module_type):
            count = count + 1
    return count


def test_mlp_batch_norm_true(device):
    num_inputs, num_outputs, hidden_size = 10, 5, [32, 64]
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        batch_norm=True,
        layer_norm=False,  # Ensure only batch_norm is tested here
        device=device,
    )

    # Check for BatchNorm1d layers
    assert count_modules_by_type(evolvable_mlp.model, torch.nn.BatchNorm1d) == len(hidden_size), "Incorrect number of BatchNorm1d layers"

    # Check order: BatchNorm1d -> Activation
    # The structure is [Linear, (BatchNorm1d), (LayerNorm), Activation, Linear, (BatchNorm1d), (LayerNorm), Activation, ..., OutputLinear, OutputActivation]
    # We expect BatchNorm1d before Activation for hidden layers
    expected_sequences = []
    for i in range(len(hidden_size)):
        # Linear -> BatchNorm1d -> Activation
        expected_sequences.append((torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.ReLU)) # Assuming ReLU is default, adjust if not

    # This check is a bit tricky due to the flexible nature of layer_norm.
    # We will iterate through the model and check the immediate successor of BatchNorm1d.
    model_layers = list(evolvable_mlp.model.children())
    bn_indices = [i for i, layer in enumerate(model_layers) if isinstance(layer, torch.nn.BatchNorm1d)]

    for bn_idx in bn_indices:
        # Ensure BN is not the last layer and is followed by an activation (or another norm if that's the design)
        assert bn_idx + 1 < len(model_layers), "BatchNorm1d should not be the last layer in a hidden block."
        # Here, we assume activation is ReLU. If other activations are used by default, this needs to be more flexible.
        assert isinstance(model_layers[bn_idx+1], (torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh, torch.nn.Sigmoid, NoisyLinear)), \
            f"BatchNorm1d at index {bn_idx} should be followed by an activation, found {type(model_layers[bn_idx+1])}"

    # Forward pass
    input_tensor = torch.randn(16, num_inputs).to(device) # Using a batch size > 1 for BN
    with torch.no_grad():
        output_tensor = evolvable_mlp.forward(input_tensor)
    assert output_tensor.shape == (16, num_outputs)
    assert evolvable_mlp.batch_norm # Check attribute

def test_mlp_batch_norm_false(device):
    num_inputs, num_outputs, hidden_size = 10, 5, [32, 64]
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        batch_norm=False,
        device=device,
    )
    assert count_modules_by_type(evolvable_mlp.model, torch.nn.BatchNorm1d) == 0, "BatchNorm1d layers should not be present"

    # Forward pass
    input_tensor = torch.randn(1, num_inputs).to(device)
    with torch.no_grad():
        output_tensor = evolvable_mlp.forward(input_tensor)
    assert output_tensor.shape == (1, num_outputs)
    assert not evolvable_mlp.batch_norm # Check attribute

def test_mlp_batch_norm_and_layer_norm(device):
    num_inputs, num_outputs, hidden_size = 10, 5, [32, 64]
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        batch_norm=True,
        layer_norm=True,
        device=device,
    )
    assert count_modules_by_type(evolvable_mlp.model, torch.nn.BatchNorm1d) == len(hidden_size)
    assert count_modules_by_type(evolvable_mlp.model, torch.nn.LayerNorm) == len(hidden_size) + evolvable_mlp.output_layernorm # +1 if output_layernorm is True

    # Check order: Linear -> BatchNorm1d -> LayerNorm -> Activation for hidden layers
    # This check needs to be more robust. Iterate and find sequences.
    model_layers = list(evolvable_mlp.model.children())
    for i in range(len(model_layers) - 3): # Iterate up to where a sequence of 4 can start
        if isinstance(model_layers[i], torch.nn.Linear) and \
           not evolvable_mlp.model[i].out_features == num_outputs: # Exclude output linear layer for this check
            # This linear layer is a hidden layer
            assert isinstance(model_layers[i+1], torch.nn.BatchNorm1d), \
                f"Linear layer at {i} should be followed by BatchNorm1d, found {type(model_layers[i+1])}"
            assert isinstance(model_layers[i+2], torch.nn.LayerNorm), \
                f"BatchNorm1d at {i+1} should be followed by LayerNorm, found {type(model_layers[i+2])}"
            assert isinstance(model_layers[i+3], (torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh, torch.nn.Sigmoid)), \
                f"LayerNorm at {i+2} should be followed by Activation, found {type(model_layers[i+3])}"


    # Forward pass
    input_tensor = torch.randn(16, num_inputs).to(device)
    with torch.no_grad():
        output_tensor = evolvable_mlp.forward(input_tensor)
    assert output_tensor.shape == (16, num_outputs)
    assert evolvable_mlp.batch_norm
    assert evolvable_mlp.layer_norm
    assert evolvable_mlp.net_config["batch_norm"]

def test_mlp_layer_norm_only(device):
    num_inputs, num_outputs, hidden_size = 10, 5, [32, 64]
    evolvable_mlp = EvolvableMLP(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        batch_norm=False,
        layer_norm=True,
        device=device,
    )
    assert count_modules_by_type(evolvable_mlp.model, torch.nn.BatchNorm1d) == 0
    assert count_modules_by_type(evolvable_mlp.model, torch.nn.LayerNorm) == len(hidden_size) + evolvable_mlp.output_layernorm

    # Check order: Linear -> LayerNorm -> Activation for hidden layers
    model_layers = list(evolvable_mlp.model.children())
    for i in range(len(model_layers) - 2):
        if isinstance(model_layers[i], torch.nn.Linear) and \
           not evolvable_mlp.model[i].out_features == num_outputs:
            assert isinstance(model_layers[i+1], torch.nn.LayerNorm), \
                f"Linear layer at {i} should be followed by LayerNorm, found {type(model_layers[i+1])}"
            assert isinstance(model_layers[i+2], (torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh, torch.nn.Sigmoid)), \
                f"LayerNorm at {i+1} should be followed by Activation, found {type(model_layers[i+2])}"

    # Forward pass
    input_tensor = torch.randn(1, num_inputs).to(device)
    with torch.no_grad():
        output_tensor = evolvable_mlp.forward(input_tensor)
    assert output_tensor.shape == (1, num_outputs)
    assert not evolvable_mlp.batch_norm
    assert evolvable_mlp.layer_norm
    assert not evolvable_mlp.net_config["batch_norm"]
