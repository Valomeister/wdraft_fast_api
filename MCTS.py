import numpy as np
import math
import encode_features
import BrawlDraft
import static_data


class PrimitiveNode:
    def __init__(self, game, args, state, parent=None, action_taken=None, moves_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.moves_count = moves_count

        self.children = []
        # [0, 0, 0, 1, 1, 0, 1, 1, 0]
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)

        relative_player = self.game.get_relative_player(self.moves_count + 1)
        relative_state = self.game.change_perspective(child_state, player=relative_player)
        child_state = relative_state

        child = PrimitiveNode(self.game, self.args, child_state, self, action, self.moves_count + 1)
        self.children.append(child)

        return child

    def simulate(self):
        rollout_state = self.state.copy()
        rollout_moves_count = self.moves_count

        while rollout_moves_count <= 5:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, 1)
            relative_player = self.game.get_relative_player(rollout_moves_count + 1)
            relative_state = self.game.change_perspective(rollout_state, player=relative_player)
            rollout_state = relative_state

            rollout_moves_count += 1

        value = self.game.get_terminal_values(rollout_state, self.args['mode_name'], self.args['map_name'])[0]
        if self.game.get_player_from_moves(self.moves_count) == -1:
            value = self.game.get_opponent_value(value)

        return value

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            relative_player = self.game.get_relative_player(self.parent.moves_count)
            if relative_player == -1:
                value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)


class PrimitiveMCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        cur_moves_count = self.game.get_moves_count(state)
        root = PrimitiveNode(self.game, self.args, state, moves_count=cur_moves_count)

        for search in range(self.args['num_searches']):
            if search != 0 and search % 1_000 == 0:
                # print(f"{search / self.args['num_searches'] * 100:.2f}%")
                pass
            node = root

            while node.is_fully_expanded():
                node = node.select()

            is_terminal = self.game.is_terminated(node.moves_count)

            if is_terminal:
                value = self.game.get_terminal_values(node.state, self.args['mode_name'], self.args['map_name'])[0]
                value = self.game.get_opponent_value(value)
            else:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs, root.value_sum / root.visit_count, root




def get_mcts_results(matches_data, n_searches, bans_mask):
    states, modes_names, maps_names = encode_features.get_states_from_matches(matches_data)

    neutral_states = encode_features.get_neutral_states(states)
    neutral_state = neutral_states[0]
    bans_mask = np.array(bans_mask)
    min_len = min(len(neutral_state), len(bans_mask))
    neutral_state[:min_len][np.array(bans_mask[:min_len]) == 1] = 2

    draft = BrawlDraft.BrawlDraft()
    args = {
        'C': 1.414,
        'num_searches': n_searches,
        'mode_name': modes_names[0],
        'map_name': maps_names[0]
    }
    mcts = PrimitiveMCTS(draft, args)
    probs, val, mcts_root = mcts.search(neutral_state)

    return probs, val

def top_n_brawlers(probs, n):
    top_indices = np.argsort(probs)[::-1][:n]

    return list(enumerate(list(map(int, (top_indices))), start=1))

if __name__ == "__main__":

    match = [{
        "mode": "gemGrab",
        "map": "Undermine",
        "teams": [
            ["DYNAMIKE"],
            ["BONNIE", "CORDELIUS"]
        ]
    }]

    probs, val = get_mcts_results(match, 5000, np.zeros(95))

    print(top_n_brawlers(probs, 5))

    print(f"\nValue позиции: {val:.4f}")