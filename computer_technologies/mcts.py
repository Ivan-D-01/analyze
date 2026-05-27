import math
import random
import numpy as np

DEBUG = False
DEBUG_1 = True
ITERATIONS = 0

class KrestiksAndNoliks:
    def __init__(self, board=None, current_player='X'):
        if board is None:
            self.board = [[' ' for _ in range(3)] for _ in range(3)]
        else:
            self.board = [row.copy() for row in board]
        self.current_player = current_player
        self.winner = None
        self.game_over = False
        self._check_game_state()

    def _check_game_state(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                self.winner = self.board[i][0]
                self.game_over = True
                return
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                self.winner = self.board[0][i]
                self.game_over = True
                return
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            self.winner = self.board[0][0]
            self.game_over = True
            return
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            self.winner = self.board[0][2]
            self.game_over = True
            return
        
        if all(cell != ' ' for row in self.board for cell in row):
            self.game_over = True
            return

    def get_possible_moves(self):
        if self.game_over:
            return []
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def make_move(self, move):
        i, j = move
        if self.board[i][j] != ' ':
            raise ValueError("эээ сюда нельзя")
        
        new_board = [row.copy() for row in self.board]
        new_board[i][j] = self.current_player
        next_player = 'O' if self.current_player == 'X' else 'X'
        return KrestiksAndNoliks(new_board, next_player)

    def simulate_random_game(self):
        current_state = self
        while not current_state.game_over:
            possible_moves = current_state.get_possible_moves()
            move = random.choice(possible_moves)
            current_state = current_state.make_move(move)
        
        if current_state.winner == 'X':
            return 1
        elif current_state.winner == 'O':
            return -1
        else:
            return 0

    def __str__(self):
        return "\n".join(["|".join(row) for row in self.board]) + "\n"


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.q = 0
        self.untried_moves = state.get_possible_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.q / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits) for child in self.children
        ]
        if DEBUG:
            print(f'pick best child {self.children[np.argmax(choices_weights)].move}')
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        move = self.untried_moves.pop()
        if DEBUG:
            print(f'expand move: {move}')
        next_state = self.state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def is_terminal(self):
        if DEBUG:
            print(f"if game is over: {self.state.game_over}")
        return self.state.game_over


class MCTS:
    def __init__(self, root_state, iterations=10, c_param=1.4):
        self.root = MCTSNode(root_state)
        self.iterations = iterations
        self.c_param = c_param

    def search(self):
        global ITERATIONS
        for _ in range(self.iterations):
            ITERATIONS += 1
            node = self._select(self.root)
            if not node.is_terminal():
                node = node.expand()
            result = self._simulate(node)
            self._backpropagate(node, result)
        
        best_child = max(self.root.children, key=lambda child: child.visits)
        if DEBUG:
            print(f'ITERATIONS {ITERATIONS}')
        return best_child.move

    def _select(self, node):
        current_node = node
        a = 0
        if not current_node.is_terminal():
            a += 1
            if not current_node.is_fully_expanded():
                return current_node
            else:
                return current_node.best_child(self.c_param)
        if DEBUG_1:
            print(f'a123123 {a}')
        return current_node

    def _simulate(self, node):
        return node.state.simulate_random_game()

    def _backpropagate(self, node, result):
        current_node = node
        while current_node is not None:
            current_node.visits += 1

            if current_node.state.current_player == 'O':
                current_node.q += result
            else:
                current_node.q -= result
            current_node = current_node.parent


def play_game_with_mcts():
    game = KrestiksAndNoliks()
    print("Начальная доска:")
    print(game)
    
    while not game.game_over:
        if game.current_player == 'X':
            print("Ваш ход (X). Введите строку и столбец (0-2), разделенные пробелом:")
            i, j = map(int, input().split())
            game = game.make_move((i, j))
        else:
            print("Компуктер думает...")
            mcts = MCTS(game)
            best_move = mcts.search()
            game = game.make_move(best_move)
            print(f"Компуктер делает ход: {best_move}")
        
        print(game)
    
    if game.winner:
        print(f"Победил {game.winner}!")
    else:
        print("Ничья!")


if __name__ == "__main__":
    play_game_with_mcts()