import pytest
import numpy as np

from game.BrawlDraft import BrawlDraft

def test_initial_state():
    draft = BrawlDraft()
    state = draft.get_initial_state()

    assert state.shape == (110,)
    assert np.all(state == 0)

def test_next_state():
    draft = BrawlDraft()
    state = draft.get_initial_state()

    state = draft.get_next_state(state, 5, 1)
    assert state[5] == 1

    state = draft.get_next_state(state, 5, -1)
    assert state[5] == -1

def test_moves_count():
    draft = BrawlDraft()
    state = draft.get_initial_state()

    state[1] = 1
    state[2] = -1

    assert draft.get_moves_count(state) == 2

def test_termination():
    draft = BrawlDraft()

    assert not draft.is_terminated(3)
    assert draft.is_terminated(6)

@pytest.mark.parametrize("move_count, expected", [
    (0, 1),
    (1, 1),
    (2, -1),
    (4, 1),
    (6, -1),
])
def test_player_from_moves(move_count, expected):
    draft = BrawlDraft()
    assert draft.get_player_from_moves(move_count) == expected

def test_change_perspective():
    draft = BrawlDraft()

    state = np.array([1, -1, 0])
    flipped = draft.change_perspective(state, -1)

    assert np.array_equal(flipped, [-1, 1, 0])

def test_encoded_state_shape():
    draft = BrawlDraft()

    state = np.zeros(110)
    mode = np.array(["knockout"])
    map_ = np.array(["Belle's Rock"])

    encoded = draft.get_encoded_state(state, mode, map_)

    assert encoded.shape == (1, draft.encoded_state_size)
    assert encoded.dtype == np.int8

def test_encoded_contains_mode_and_map():
    draft = BrawlDraft()

    state = np.zeros(110)
    encoded = draft.get_encoded_state(
        state,
        np.array(["knockout"]),
        np.array(["Belle's Rock"])
    )[0]

    assert np.sum(encoded[:10]) == 1
    assert np.sum(encoded[10:60]) == 1

def test_terminal_values_shape():
    draft = BrawlDraft()

    state = np.zeros(110)
    mode = np.array(["knockout"])
    map_ = np.array(["Belle's Rock"])

    values = draft.get_terminal_values(state, mode, map_)

    assert isinstance(values, np.ndarray)
    assert values.shape == (1, )
    assert -1 <= values <= 1