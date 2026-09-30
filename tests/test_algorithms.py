"""Small examples of Bellman updates and discretized dynamics."""

import numpy as np
import pytest

from algorithms import build_transition_matrix, value_iteration


def test_value_iteration_solves_a_small_graph_in_multiple_dimensions():
    transition_matrix = np.array([[[4, 1], [4, 4]], [[0, 4], [3, 4]]])
    reward = np.array([[[1, 0], [4, 0]], [[0, 1], [1, 5]]])
    expected_value = np.array([[2, 4], [1, 5]])
    expected_policy = np.array([[1, 0], [0, 1]])

    # Node 4 is terminal. At node 2, both actions have value 1; choose the first.
    for grid_shape in [(2, 2), (2, 1, 2)]:
        table_shape = (*grid_shape, 2)
        value, policy = value_iteration(
            transition_matrix.reshape(table_shape),
            reward.reshape(table_shape),
            discount=0.5,
        )

        assert value.shape == policy.shape == grid_shape
        assert value == pytest.approx(expected_value.reshape(grid_shape))
        assert policy.tolist() == expected_policy.reshape(grid_shape).tolist()


def test_discounted_self_loop_and_iteration_limit():
    transition_matrix = np.array([[0]])
    reward = np.array([[1.0]])

    value, policy = value_iteration(transition_matrix, reward, discount=0.9)

    # 1 + 0.9 + 0.9**2 + ... = 1 / (1 - 0.9).
    assert value[0] == pytest.approx(10.0)
    assert policy[0] == 0
    value, _ = value_iteration(
        transition_matrix, reward, discount=0.9, max_iterations=2
    )
    assert value[0] == pytest.approx(1.9)  # 1 + 0.9 after two sweeps


def test_transition_matrix_uses_complete_coordinates_and_flat_node_ids():
    grid_points = np.array(
        [[[4, 0], [0, 0], [0, 4]], [[4, 4], [1, 3], [3, 2]]], dtype=float
    )
    actions = np.array([[0.0, 0.0], [-2.0, 0.0], [-2.7, 1.3]])

    def step(state, action):
        return state + action

    transition_matrix = build_transition_matrix(grid_points, actions, step)

    assert transition_matrix.shape == (2, 3, 3)
    assert transition_matrix[:, :, 0].tolist() == [[0, 1, 2], [3, 4, 5]]
    # [2, 0] is halfway between nodes 0 and 1. The first stored node wins,
    # even though it has the larger coordinate.
    assert transition_matrix[0, 0, 1] == 0
    # [1.3, 1.3] is closest to node 4: [1, 3]. Rounding each coordinate
    # independently would instead select [1, 2], which is not a grid point.
    assert transition_matrix[0, 0, 2] == 4


def test_transition_matrix_distinguishes_failure_from_boundary_nodes():
    grid_points = np.array([[0.0], [1.0], [3.0]])
    original_points = grid_points.copy()
    actions = [0, 1, 2, 3]

    def step(state, action):
        if action == 0:
            return None
        if action == 1:
            return np.array([4.0])
        # A rollout may update its state in place; this must not alter the grid.
        state[0] = 0.0 if action == 2 else 3.0
        return state

    transition_matrix = build_transition_matrix(grid_points, actions, step)

    # Terminal node 3 is distinct from physical node 0. Both endpoints are valid.
    assert transition_matrix.tolist() == [[3, 3, 0, 2]] * 3
    assert grid_points.tolist() == original_points.tolist()
