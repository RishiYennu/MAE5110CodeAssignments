import inspect

import numpy as np

import models


def test_model_interface():
    for name, model in inspect.getmembers(models, inspect.ismodule):
        assert callable(getattr(model, "generate_params", None)), name
        assert callable(getattr(model, "dynamics", None)), name

        params = model.generate_params()
        state = np.array([0.1, 0.2])
        derivative = model.dynamics(0.0, state, params)

        assert np.shape(derivative) == state.shape, name
