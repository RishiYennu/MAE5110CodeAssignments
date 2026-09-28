"""Discounted value iteration for deterministic transitions on a state grid."""

import numpy as np


def value_iteration(
    transition_matrix, reward, discount=0.95, tolerance=1e-8, max_iterations=10_000
):
    """Return state-value estimates and greedy action indices.

    Both tables have shape (*grid_shape, number_of_actions). Successors are
    C-order node IDs; ID prod(grid_shape) is terminal with continuation zero.
    Sweeps are synchronous, and the first action wins an exact value tie.
    Return the latest estimate if max_iterations is reached.
    """
    transition_matrix = np.asarray(transition_matrix)
    reward = np.asarray(reward, dtype=float)
    grid_shape = transition_matrix.shape[:-1]
    value = np.zeros(grid_shape)
    for _ in range(max_iterations):
        previous = value.copy()
        continuation = np.append(previous.ravel(), 0.0)
        action_values = reward + discount * continuation[transition_matrix]
        value = np.max(action_values, axis=-1)
        if np.max(np.abs(value - previous)) < tolerance:
            break

    # Choose actions using the returned values, rather than the previous sweep.
    continuation = np.append(value.ravel(), 0.0)
    action_values = reward + discount * continuation[transition_matrix]
    policy = np.argmax(action_values, axis=-1)
    return value, policy
