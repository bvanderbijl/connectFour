import os
import numpy as np
from math import exp, log

import random

from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers.legacy import Adam


class RandomConnectPlayer:
    def __init__(self, player_number, agent):
        self.player_number = player_number
        self.agent = agent
    
    def make_move(self):
        moves = self.agent.get_valid_moves()
        return random.choice(moves)


class DQNAgent:

    def __init__(self, player_number, state_shape, action_size, episodes=1000, gamma=0.9, epsilon=0.10, epsilon_min=0.01, model_name=None):
        self.player_number = player_number
        
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # initial exploration rate
        self.epsilon_min = epsilon_min
        if self.epsilon > 0: 
            self.epsilon_decay = exp((log(self.epsilon_min) - log(self.epsilon))/(0.8*episodes)) # reaches epsilon_min after 80% of iterations
        else:
            self.epsilon_decay = 0
        
        if model_name != None:
            self.load(model_name)
        else:
            self.model = self._build_model()
     
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(50, input_shape=self.state_shape, activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate = 0.00001))
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def make_move(self, agent):
        valid_moves = agent.get_valid_moves()
        board = agent.board.copy()
        board = self.normalize_board(board)
        state = np.reshape(board, (1,-1))
        
        if agent.current_player != self.player_number:
            state = state * -1
        
        if np.random.rand() <= self.epsilon: # Exploration
            return random.choice(valid_moves)
        
        act_values = self.model.predict(state, verbose=0) # Exploitation
        action = np.argmax([ prediction if i+1 in valid_moves else -np.inf for i, prediction in enumerate(act_values[0])])

        return action + 1
    
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_shape[0]), dtype=int)
        actions = np.zeros((batch_size, ), dtype=int)
        rewards = np.zeros((batch_size, ), dtype=float)
        next_states = np.zeros((batch_size, self.state_shape[0]), dtype=int)
        dones = np.zeros((batch_size, ), dtype=bool)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state[0]
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state[0]
            dones[i] = done
        
        q_eval = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        
        q_target = q_eval.copy()
        action_index = actions - 1
        batch_index = np.arange(batch_size)
        
        q_target[batch_index, action_index] = rewards + ((self.gamma * np.max(q_next, axis=1)) * ~dones)
        
        _ = self.model.fit(states, q_target, verbose=0)
        
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def load(self, name):
        try:
            self.model = load_model(os.path.join('..', 'models', name))
        except:
            self.model = self._build_model()
    
    def save(self, name):
        self.model.save(os.path.join('..', 'models', name), save_format='h5')
    
    def normalize_board(self, board: np.array):
        normalized_board = board.copy()
        
        for col_index, col_value in enumerate(board):
            for row_index, row_value in enumerate(col_value):
                if row_value == -1:
                    normalized_board[col_index,row_index] = 0
                elif row_value == self.player_number:
                    normalized_board[col_index,row_index] = 1
                else:
                    normalized_board[col_index,row_index] = -1
        return normalized_board
    
    def switch_side(self):
        self.player_number = (self.player_number + 1) % 2