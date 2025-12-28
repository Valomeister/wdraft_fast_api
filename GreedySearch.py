import time

import numpy as np
import torch
from torch import nn

import encode_features
import static_data
from BrawlDraft import BrawlDraft
from ValueNetwork import ValueNetwork

game = BrawlDraft()
input_size = game.encoded_state_size


model = ValueNetwork(input_size)
model.load_state_dict(torch.load("models/ValueNetwork/value_network_best.pt", map_location="cpu"))
model.eval()

def greedy_search(state, mode_name, map_name):
    print(game.get_concise_state(state))
    move_count = game.get_moves_count(state)
    valid_moves = sorted(game.get_valid_moves(state))
    encoded_states = []
    for i in range(len(valid_moves)):
        next_state = game.get_next_state(state.copy(), valid_moves[i], 1)
        next_rel_player = game.get_relative_player(move_count + 1)
        next_rel_state = game.change_perspective(next_state, player=next_rel_player)

        encoded_state = game.get_encoded_state(next_rel_state, mode_name, map_name)
        encoded_states.append((game.get_concise_state(next_rel_state), encoded_state))

        if move_count < 5:
            for j in range(len(valid_moves)):
                if i == j:
                    continue
                next_next_state = game.get_next_state(next_rel_state.copy(), valid_moves[j], 1)
                next_next_rel_player = game.get_relative_player(move_count + 2)
                next_next_rel_state = game.change_perspective(next_next_state, player=next_next_rel_player)

                encoded_state = game.get_encoded_state(next_next_rel_state, mode_name, map_name)
                encoded_states.append((game.get_concise_state(next_next_rel_state), encoded_state))

    only_encoded_states = [i[1] for i in encoded_states]
    tensor = torch.tensor(np.array(only_encoded_states), dtype=torch.float)
    with torch.no_grad():
        values = model(tensor)
    values = values.squeeze(-1).cpu().numpy()

    results = np.zeros(game.action_size, dtype=np.float32)
    for i in range(len(valid_moves)):
        if move_count < 5:
            next_value = values[i * (len(valid_moves))]
        else:
            next_value = values[i]
        if game.get_player_from_moves(move_count + 2) != game.get_player_from_moves(move_count + 1):
            next_value *= -1
        results[valid_moves[i]] += next_value / len(valid_moves)
        if move_count < 5:
            for j in range(len(valid_moves)):
                if i == j:
                    continue
                next_next_value = values[i * (len(valid_moves)) + j]
                if game.get_player_from_moves(move_count + 3) != game.get_player_from_moves(move_count + 1):
                    next_next_value *= -1
                results[valid_moves[i]] += next_next_value / len(valid_moves)

    action_probs = np.zeros(game.action_size)
    for i in range(len(valid_moves)):
        action_probs[valid_moves[i]] = results[valid_moves[i]]

    return {
        "action_probs": action_probs_to_dict(action_probs),
        "max_action_probs": max(action_probs),
        "avg_value": float(np.mean(values)),
    }

def action_probs_to_dict(action_probs):
    probs_list = action_probs.tolist() if hasattr(action_probs, "tolist") else list(action_probs)
    result = {static_data.BRAWLERS[i]: probs_list[i] for i in range(len(probs_list)) if i < len(static_data.BRAWLERS)}

    return result

def get_greedy_search_results(match_data, bans_mask):
    state_from_match = encode_features.get_state_from_match(match_data)
    if state_from_match:
        state, mode_name, map_name = state_from_match

    assert state_from_match
    neutral_state = state
    min_len = min(len(neutral_state), len(bans_mask))
    neutral_state[:min_len][bans_mask[:min_len] == 1] = 0

    return greedy_search(neutral_state, mode_name, map_name)

if __name__ == "__main__":
    # s0 = game.get_initial_state()
    # game.get_next_state(s0, 0, 1)
    # game.get_next_state(s0, 2, -1)
    # greedy_search(s0, "gemGrab", "Hard Rock Mine")

    X = torch.randn(100**2, 390)

    start = time.perf_counter()

    with torch.no_grad():
        y = model(X)

    end = time.perf_counter()

    print(f"Inference time: {end - start:.6f} sec")

