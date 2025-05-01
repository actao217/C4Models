import math
import random

class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.visits = 0
        self.score = 0
        self.children = None

    def is_leaf(self):
        return self.children is None or self.children == []

    def is_terminal(self):
        return self.children == []

class Tree:
    def __init__(self):
        self.root = Node(None, None)

class MctsStrategy:
    def __init__(self, rollout_limit):
        self.rollout_limit = rollout_limit

    def move(self, game, player_id):
        tree = Tree()
        for _ in range(self.rollout_limit):
            self.__simulate_game(game.clone(), tree, player_id)
        best_child = max(tree.root.children, key=lambda c: c.visits)
        return best_child.action[1]

    def __simulate_game(self, game, tree, player_id):
        current_node = tree.root
        nodes_to_update = [current_node]

        while not current_node.is_leaf():
            current_node = self.__get_child_with_highest_ucb(current_node)
            move_player, move = current_node.action
            game.current_player = move_player
            game.move(move)
            nodes_to_update.append(current_node)

        if not game.finished:
            next_player = game.current_player
            self.__add_children(current_node, game.board, next_player)
            best_new_child = self.__get_child_with_highest_ucb(current_node)
            game.current_player = best_new_child.action[0]
            game.move(best_new_child.action[1])
            nodes_to_update.append(best_new_child)

            while not game.finished:
                valid = game.board.get_valid_moves()
                move = random.choice(valid)
                game.move(move)

        score = 1 if game.winner == player_id else 0 if game.winner is None else -1
        for node in nodes_to_update:
            node_score = score if node.action is None or node.action[0] == player_id else -score
            node.visits += 1
            node.score += node_score

    def __add_children(self, parent_node, board, player_id):
        moves = board.get_valid_moves()
        parent_node.children = [Node(parent_node, (player_id, move)) for move in moves]

    def __get_child_with_highest_ucb(self, node):
        max_ucb = -math.inf
        best_children = []
        for child in node.children:
            ucb_val = self.__ucb(child)
            if ucb_val > max_ucb:
                best_children = [child]
                max_ucb = ucb_val
            elif ucb_val == max_ucb:
                best_children.append(child)
        return random.choice(best_children)

    def __ucb(self, node):
        if node.visits == 0:
            return math.inf
        return node.score / node.visits + 2 * math.sqrt(math.log(node.parent.visits) / node.visits)
