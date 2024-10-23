import random
import numpy as np

# Define constants
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
BOARD_SIZE = 3

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = PLAYER_X

    def reset(self):
        self.board.fill(0)
        self.current_player = PLAYER_X

    def available_moves(self):
        return list(zip(*np.where(self.board == EMPTY)))

    def play_move(self, row, col):
        if self.board[row, col] == EMPTY:
            self.board[row, col] = self.current_player
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
            return True
        return False

    def check_winner(self):
        for player in [PLAYER_X, PLAYER_O]:
            # Check rows, columns and diagonals
            if any(np.all(self.board[i, :] == player) for i in range(BOARD_SIZE)) or \
               any(np.all(self.board[:, j] == player) for j in range(BOARD_SIZE)) or \
               np.all(np.diagonal(self.board) == player) or \
               np.all(np.diagonal(np.fliplr(self.board)) == player):
                return player
        if np.all(self.board != EMPTY):
            return -1  # Draw
        return None  # Game ongoing

class MENACE:
    def __init__(self):
        self.states = {}
        self.learning_rate = 0.1
        self.epsilon = 0.1

    def get_state_key(self, board):
        return tuple(board.flatten())

    def choose_move(self, board):
        state_key = self.get_state_key(board)
        if state_key not in self.states:
            self.states[state_key] = {move: 0 for move in self.available_moves(board)}

        if random.random() < self.epsilon:
            return random.choice(self.available_moves(board))  # Explore

        # Exploit: Choose the best move based on current estimates
        return max(self.states[state_key], key=self.states[state_key].get)

    def update(self, board, move, reward):
        state_key = self.get_state_key(board)
        if state_key in self.states:
            self.states[state_key][move] += self.learning_rate * (reward - self.states[state_key][move])

    def available_moves(self, board):
        return list(zip(*np.where(board == EMPTY)))

def play_game(menace):
    game = TicTacToe()
    board_state_history = []
    moves = []

    while True:
        board_state_history.append(game.board.copy())
        move = menace.choose_move(game.board)
        moves.append(move)
        game.play_move(move[0], move[1])
        winner = game.check_winner()

        if winner is not None:
            # Update MENACE with the result
            reward = 1 if winner == PLAYER_X else -1 if winner == PLAYER_O else 0
            for state, move in zip(board_state_history, moves):
                menace.update(state, move, reward)
            break

if __name__ == "__main__":
    menace = MENACE()

    # Play multiple games for training
    for _ in range(10000):  # Number of training games
        play_game(menace)

    # Display learned state values
    for state_key, moves in menace.states.items():
        print("State:", np.array(state_key).reshape(3, 3), " -> Move values:", moves)
