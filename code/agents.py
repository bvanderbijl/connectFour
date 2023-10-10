import numpy as np
from math import exp, log

import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam


class RandomConnectPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
    
    def make_move(self, agent):
        moves = agent.get_valid_moves()
        return random.choice(moves)


class DQNAgent:

    def __init__(self, state_shape, action_size, episodes, model_name=None):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = 0.9   # discount rate
        self.epsilon = 0.10  # initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exp((log(self.epsilon_min) - log(self.epsilon))/(0.8*episodes)) # reaches epsilon_min after 80% of iterations
        self.model = self._build_model()
        
        if model_name != None:
            self.load(model_name)
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_shape=self.state_shape, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate = 0.00001))
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def make_move(self, user, state, valid_moves):
        if np.random.rand() <= self.epsilon: # Exploration
            return random.choice(valid_moves)
        act_values = self.model.predict(state, verbose=0) # Exploitation
        if user == 0:
            action = np.argmax([ prediction if i+1 in valid_moves else -np.inf for i, prediction in enumerate(act_values[0])]) + 1
        else:
            action = np.argmin([prediction if i+1 in valid_moves else np.inf for i, prediction in enumerate(act_values[0])]) + 1
        return action
    
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_shape[0]), dtype=int)
        actions = np.zeros((batch_size, 1), dtype=int)
        rewards = np.zeros((batch_size, 1), dtype=float)
        next_states = np.zeros((batch_size, self.state_shape[0]), dtype=int)
        dones = np.zeros((batch_size, 1))
        
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
        
        q_target[batch_index, action_index] = rewards + self.gamma*np.min(q_next, axis=1)*dones
        
        _ = self.model.fit(states, q_target, verbose=0)
        
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)