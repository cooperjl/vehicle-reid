import json
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

from vehicle_reid.utils import AverageMeter, NumpyEncoder, load_weights, pad_label


def test_pad_label():
    """Test the pad_label util for multiple datasets."""
    assert pad_label(15, "veri") == "015"
    assert pad_label(15, "vric") == "0015"

    with pytest.raises(ValueError):
        pad_label(15, "invalid")


def test_avg_meter():
    """Test the AverageMeter class."""
    meter = AverageMeter()
    meter.update(5)
    assert meter.sum == 5
    assert meter.avg == 5
    meter.update(2)
    assert meter.sum == 7
    assert meter.avg == 3.5


def test_numpy_encoder():
    """Test that the NumpyEncoder class correctly deals with numpy arrays."""
    array = np.arange(5)
    dump = json.dumps(array, cls=NumpyEncoder)

    assert np.array_equal(json.loads(dump), array)


def test_load_weights():
    """Test that the load_weights function correctly loads the state_dict into a model."""
    model = nn.Sequential(nn.Linear(2, 1), nn.ReLU())

    state_dict = OrderedDict(
        {"0.weight": torch.tensor([[-0.5, -0.05]]), "0.bias": torch.tensor([-0.3])}
    )
    load_weights(model, state_dict)

    updated_dict = model.state_dict()

    assert all(torch.equal(v, updated_dict[k]) for k, v in state_dict.items())


def test_unequal_load_weights():
    """Test that the load_weights function correctly deals with unequally sized weights."""
    model = nn.Sequential(nn.Linear(2, 1), nn.ReLU(), nn.Linear(1, 1))

    state_dict = OrderedDict(
        {
            "0.weight": torch.tensor([[-0.5, -0.05]]),
            "0.bias": torch.tensor([-0.3]),
            "2.weight": torch.tensor([[0.5, 0.05]]),
            "2.bias": torch.tensor([-0.7]),
        }
    )
    load_weights(model, state_dict)

    updated_dict = model.state_dict()

    # Expect that since the size of 2.weight in state_dict is wrong, it should not be updated, but the rest should be
    assert all(
        torch.equal(v, updated_dict[k])
        for k, v in state_dict.items()
        if k != "2.weight"
    )
    assert not torch.equal(updated_dict["2.weight"], state_dict["2.weight"])
