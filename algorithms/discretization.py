"""Build a deterministic transition table from continuous state rollouts."""

import numpy as np


def build_transition_matrix(grid_points, actions, step):
    """Map each grid point/action to its nearest successor node in C order.

    grid_points has shape (*grid_shape, state_dimension). The returned table
    has shape (*grid_shape, number_of_actions). step(state, action) returns a
    state vector, or None for failure. Failures and states outside the grid's
    coordinate bounding box map to terminal ID prod(grid_shape).

    Distances use the complete state vectors and their supplied coordinate
    units. The first stored point wins an exact squared-distance tie.
    """
    grid_points = np.asarray(grid_points, dtype=float)
    actions = np.asarray(actions)
    grid_shape = grid_points.shape[:-1]
    state_dimension = grid_points.shape[-1]
    points = grid_points.reshape(-1, state_dimension)
    terminal = len(points)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    transitions = np.full((terminal, len(actions)), terminal, dtype=int)

    for node, state in enumerate(points):
        for action_index, action in enumerate(actions):
            next_state = step(state.copy(), action)
            if next_state is None:
                continue
            next_state = np.asarray(next_state, dtype=float)
            if np.any(next_state < lower) or np.any(next_state > upper):
                continue

            distances_squared = np.sum((points - next_state) ** 2, axis=1)
            transitions[node, action_index] = np.argmin(distances_squared)

    return transitions.reshape(grid_shape + (len(actions),))
