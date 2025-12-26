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
    move_count = game.get_moves_count(state)
    valid_moves = sorted(game.get_valid_moves(state))
    results = {move: {} for move in valid_moves}
    for move in valid_moves:
        next_state = game.get_next_state(state.copy(), move, 1)
        rel_player = game.get_relative_player(move_count + 1)
        rel_state = game.change_perspective(next_state, player=rel_player)
        results[move]["child_state"] = game.get_concise_state(rel_state)

        encoded_state = game.get_encoded_state(rel_state, mode_name, map_name)
        results[move]["encoded_child_state"] = encoded_state

    encoded_child_states = [results[move]["encoded_child_state"] for move in valid_moves]
    tensor = torch.tensor(np.array(encoded_child_states), dtype=torch.float)

    with torch.no_grad():
        values = model(tensor)

    values = values.squeeze(-1).cpu().numpy()
    if game.get_relative_player(move_count + 1) == -1:
        values *= -1

    for i in range(len(values)):
        results[valid_moves[i]]["value"] = (values[i] + 1) / 2

    action_probs = np.zeros(game.action_size)
    for move in valid_moves:
        action_probs[move] = results[move]["value"]
    action_probs /= np.sum(action_probs)

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
    neutral_state = state # получаем от клиента сразу neutral
    min_len = min(len(neutral_state), len(bans_mask))
    neutral_state[:min_len][bans_mask[:min_len] == 1] = 2

    return greedy_search(neutral_state, mode_name, map_name)

if __name__ == "__main__":
    s0 = game.get_initial_state()
    game.get_next_state(s0, 0, 1)
    game.get_next_state(s0, 2, -1)
    greedy_search(s0, "gemGrab", "Hard Rock Mine")
