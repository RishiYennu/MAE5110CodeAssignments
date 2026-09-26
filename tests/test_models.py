import inspect

import numpy as np

import models


def test_model_interface():
    for name, model in inspect.getmembers(models, inspect.ismodule):
        assert callable(getattr(model, "generate_params", None)), name
        assert callable(getattr(model, "generate_initial_condition", None)), name
        assert callable(getattr(model, "dynamics", None)), name

        params = model.generate_params()
        state = model.generate_initial_condition()
        assert np.ndim(state) == 1, name
        derivative = model.dynamics(0.0, state, params)

        assert np.shape(derivative) == np.shape(state), name
