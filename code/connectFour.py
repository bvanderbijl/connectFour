import numpy as np
import gym

ROW_COUNT = 6
COLUMN_COUNT = 7

class connectFourAgent:
    def __init__(self):
        self.row_count = ROW_COUNT
        self.column_count = COLUMN_COUNT
        
        self.current_player = 0
        self.board = self.create_board()
        
        self.game_over = False
        self.turns = 0
        
    def next_move(self):
        return (self.current_player + 1) % 2
    
    def create_board(self):
        board = np.full((self.row_count, self.column_count), -1)
        return board
        
    def drop_piece(self, col):
        column = col-1
        if self.is_valid_move(column):
            row = self.get_open_row(column)
            self.board[row, column] = self.current_player
            
            self.current_player = self.next_move()
            self.turns += 1
        else:
            print(f"The selected move is invalid! The column is full")
    
    def get_open_row(self, col):
        return np.min([i for i, x in enumerate(self.board[:, col]) if x == -1])
    
    def print_board(self):
        print(np.flip(self.board, 0))
    
    def is_valid_move(self, col):
        if col <= self.column_count:
            return self.board[self.row_count-1, col] == -1
        return False
    
    def get_valid_moves(self):
        return [i + 1 for i, x in enumerate(self.board[self.row_count-1]) if x == -1]
    
    def winning_move(self, player):
        # Check horizontal locations for win
        for c in range(self.column_count-3):
            for r in range(self.row_count):
                if self.board[r][c] == player and self.board[r][c+1] == player and self.board[r][c+2] == player and self.board[r][c+3] == player:
                    return True
    
        # Check vertical locations for win
        for c in range(self.column_count):
            for r in range(self.row_count-3):
                if self.board[r][c] == player and self.board[r+1][c] == player and self.board[r+2][c] == player and self.board[r+3][c] == player:
                    return True
    
        # Check positively sloped diaganols
        for c in range(self.column_count-3):
            for r in range(self.row_count-3):
                if self.board[r][c] == player and self.board[r+1][c+1] == player and self.board[r+2][c+2] == player and self.board[r+3][c+3] == player:
                    return True
    
        # Check negatively sloped diaganols
        for c in range(self.column_count-3):
            for r in range(3, self.row_count):
                if self.board[r][c] == player and self.board[r-1][c+1] == player and self.board[r-2][c+2] == player and self.board[r-3][c+3] == player:
                    return True
        
        # else game is not finished
        return False
    
    def is_draw(self):
        return len(self.get_valid_moves()) == 0
    
    def game_finished(self):
        if self.winning_move(0):
            return True
        
        if self.winning_move(1):
            return True
        
        if self.is_draw():
            return True
        
        return False


class ConnectFourGymEnv(gym.Env):
    def __init__(self):
        self.agent = connectFourAgent()
        self.action_space = self.agent.get_valid_moves()
        self.observation_space = self.agent.board.copy()
        self.reset()

    def reset(self):
        self.current_player = 0
        self.board = self.agent.create_board().copy()
        self.game_over = False
        self.turns = 0
        
        return self.board.flatten()
    
    def step(self, action):
        # Execute the action in the game
        self.agent.drop_piece(action)
        self.board = self.agent.board.copy()

        done = self.agent.game_finished()
        if done:
            if self.agent.winning_move(0):
                reward = 1.0  # Player 0 wins
            elif self.agent.winning_move(1):
                reward = -1.0  # Player 1 wins
            else:
                reward = 0.0  # It's a draw
        else:
            reward = 0.0

        info = {}  # You can add any additional info here
        return self.board.flatten(), reward, done, info
    
    def render(self):
        self.agent.print_board()