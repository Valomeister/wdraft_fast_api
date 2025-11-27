import numpy as np
from BrawlDraft import BrawlDraft
import static_data


def get_encoded_state(states, mode_names, map_names):
    encoded_state_size = static_data.MODE_LEN + static_data.MAP_LEN + static_data.BRAWLER_LEN * 3

    assert states.ndim in [1, 2]
    assert mode_names.ndim in [1, 2]
    assert map_names.ndim in [1, 2]

    if states.ndim == 1:
        states = np.expand_dims(states, axis=0)

    batch_size = states.shape[0]
    one_hots = np.zeros((batch_size, encoded_state_size), dtype=np.float32)
    for i in range(batch_size):
        state, mode_name, map_name = states[i], mode_names[i], map_names[i]

        if not mode_name in static_data.MODES or not map_name in static_data.MAPS:
            continue

        mode_vector = np.zeros(static_data.MODE_LEN, dtype=np.float32)
        mode_idx = static_data.MODES.index(mode_name)
        mode_vector[mode_idx] = 1

        map_vector = np.zeros(static_data.MAP_LEN, dtype=np.float32)
        map_idx = static_data.MAPS.index(map_name)
        map_vector[map_idx] = 1

        encoded_state = np.concatenate(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        one_hot = np.concatenate([mode_vector, map_vector, encoded_state])
        one_hots[i] = one_hot
    return one_hots

def get_states_from_matches(matches_data):
    num_batches = len(matches_data)

    states = np.zeros((num_batches, static_data.BRAWLER_LEN))
    mode_names = np.empty(num_batches, dtype='U32')
    map_names = np.empty(num_batches, dtype='U32')

    for i, match in enumerate(matches_data):
        mode_name = match['mode']
        map_name = match['map']
        result = match.get('result')
        teams = match['teams']

        if not mode_name in static_data.MODES or not map_name in static_data.MAPS:
            continue

        team1_brawlers = teams[0]
        team2_brawlers = teams[1]

        # --- Кодирование Team 1 (Multi-Hot) ---
        team1_vector = np.zeros(static_data.BRAWLER_LEN, dtype=np.int8)
        for brawler in team1_brawlers:
            brawler_idx = static_data.BRAWLERS.index(brawler)
            team1_vector[brawler_idx] = 1

        # --- Кодирование Team 2 (Multi-Hot) ---
        team2_vector = np.zeros(static_data.BRAWLER_LEN, dtype=np.int8)
        for brawler in team2_brawlers:
            brawler_idx = static_data.BRAWLERS.index(brawler)
            team2_vector[brawler_idx] = -1

        # [1, 0, 0] + [0, 0, -1] => [1, 0, -1]
        state = team1_vector + team2_vector

        states[i] = state
        mode_names[i] = mode_name
        map_names[i] = map_name

    return states, mode_names, map_names

def get_neutral_states(states):
    states_copy = np.zeros_like(states)
    draft = BrawlDraft()

    for i, state in enumerate(states):
        moves_count = draft.get_moves_count(state)
        next_player = draft.get_player_from_moves(moves_count + 1)
        relative_state = draft.change_perspective(state, player=next_player)
        states_copy[i] = relative_state

    return states_copy

def encode_features_to_onehot(matches_data):
    states, mode_names, map_names = get_states_from_matches(matches_data)

    neutral_states = get_neutral_states(states)

    X = get_encoded_state(neutral_states, mode_names, map_names)

    return X


