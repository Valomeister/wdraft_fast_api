"""
This module implements the core logic (simulation of logic) of drafting in Brawl Stars
"""
import numpy as np
import torch
from torch import nn

from . import static_data


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Размер входа должен совпадать с X.shape[1]
input_size = static_data.MODE_LEN + static_data.MAP_LEN + 3 * static_data.BRAWLER_LEN
model = SimpleNN(input_size)

MODEL_PATH = "models/draft_evaluator.pt"
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

class BrawlDraft:
    def __init__(self):
        self.mode_count = static_data.MODE_LEN
        self.map_count = static_data.MAP_LEN
        self.brawler_count = static_data.BRAWLER_LEN

        assert self.mode_count == 10
        assert self.map_count == 50
        assert self.brawler_count == 110

        self.action_size = self.brawler_count
        self.encoded_state_size = self.mode_count + self.map_count + self.brawler_count * 3
        self.game_name = "WDraft"

    def get_initial_state(self):
        return np.zeros(self.brawler_count, dtype=np.int8)

    def get_next_state(self, state, action, player):
        state[action] = player
        return state

    def get_valid_moves(self, state):
        actual_brawler_count = len(static_data.BRAWLERS)
        valid_moves_mask = (state == 0).astype(np.int8)
        valid_moves_mask[actual_brawler_count:] = 0
        valid_moves_indices = np.flatnonzero(valid_moves_mask).tolist()
        np.random.shuffle(valid_moves_indices)
        return valid_moves_indices

    def is_terminated(self, move_count):
        assert 0 <= move_count <= 6
        return move_count == 6

    def get_encoded_state(self, states, mode_names, map_names):
        # print(states.ndim, mode_names.ndim, map_names.ndim)
        assert states.ndim in [1, 2]
        if isinstance(mode_names, np.str_):
            mode_names = str(mode_names)
        if isinstance(map_names, np.str_):
            map_names = str(map_names)
             
        assert type(mode_names) == str or mode_names.ndim in [1, 2]
        assert type(map_names) == str or map_names.ndim in [1, 2]

        # Preprocessing in case of single inputs
        if type(mode_names) == str:
            mode_names = np.stack([mode_names])
        if type(map_names) == str:
            map_names = np.stack([map_names])

        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        if mode_names.ndim == 0:
                    states = np.expand_dims(states, axis=0)
        if map_names.ndim == 0:
                    states = np.expand_dims(states, axis=0)

        batch_size = states.shape[0]
        one_hots = np.zeros((batch_size, self.encoded_state_size), dtype=np.int8)
        for i in range(batch_size):
            state, mode_name, map_name = states[i], mode_names[i], map_names[i]

            mode_vector = np.zeros(self.mode_count, dtype=np.int8)
            mode_idx = static_data.MODES.index(mode_name)
            mode_vector[mode_idx] = 1

            map_vector = np.zeros(self.map_count, dtype=np.int8)
            map_idx = static_data.MAPS.index(map_name)
            map_vector[map_idx] = 1

            encoded_state = np.concatenate(
                (state == -1, (state == 0) | (state == 2), state == 1)
            ).astype(np.int8)

            one_hot = np.concatenate([mode_vector, map_vector, encoded_state])
            one_hots[i] = one_hot
        return one_hots

    def get_terminal_values(self, states, mode_names, map_names):
        x = self.get_encoded_state(states, mode_names, map_names)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(x_tensor)

        values = prediction.squeeze(-1).cpu().numpy()
        values = values * 2 - 1

        return values

    def get_player_from_moves(self, move_count):
        # get player who made last move
        assert 0 <= move_count <= 7
        return 1 if move_count in [0, 1, 4, 5] else -1

    def get_relative_player(self, move_count):
        # next player relative to current
        current_player = self.get_player_from_moves(move_count)
        next_player = self.get_player_from_moves(move_count + 1)
        return 1 if current_player == next_player else -1

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def print_readable_state(self, state):
        idx_team_a_picks = np.where(state == 1)[0]
        idx_team_b_picks = np.where(state == -1)[0]

        team_a = [static_data.BRAWLERS[i] for i in idx_team_a_picks]
        team_b = [static_data.BRAWLERS[i] for i in idx_team_b_picks]

        print(f"team a: {team_a}")
        print(f"team b: {team_b}")

    def get_concise_state(self, state):
        state_concise = [np.where(state == 1)[0].tolist(), np.where(state == -1)[0].tolist()]
        return state_concise

    def get_moves_count(self, state):
        return np.sum(state == 1) + np.sum(state == -1)

    def get_neutral_state(self, state, moves_count):
        next_player = self.get_player_from_moves(moves_count + 1)
        neutral_state = self.change_perspective(state, next_player)

        return neutral_state

if __name__ == "__main__":
    # Just random testing


    draft = BrawlDraft()

    # state = np.zeros(94)
    # state[0:3] = 1
    # state[4:7] = -1
    # print(draft.get_encoded_state(state, np.array(["bounty"]), np.array(["Shooting Star"])))
    # print("\n" * 10)

    move_count = 0
    player = draft.get_player_from_moves(move_count + 1)
    state = np.array(draft.get_initial_state())
    # mode_name = np.array([random.choice(static_data.MODES)])
    # map_name = np.array([random.choice(static_data.MAPS_FOR_MODES[mode_name[0]])])
    mode_name = np.array(['knockout'])
    map_name = np.array(["Belle's Rock"])

    state = draft.get_next_state(state, 6, 1)
    state = draft.get_next_state(state, 9, 1)
    state = draft.get_next_state(state, 22, 1)
    state = draft.get_next_state(state, 11, -1)
    state = draft.get_next_state(state, 87, -1)
    state = draft.get_next_state(state, 94, -1)

    print(state.ndim, mode_name.ndim, map_name.ndim)
    print(state)
    print(draft.get_encoded_state(state, mode_name, map_name))
    print(draft.get_encoded_state(state, mode_name, map_name).shape)
    print(draft.get_terminal_values(state, mode_name, map_name))