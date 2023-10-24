import numpy as np

from connectFour import *
from connectFourGame import *
from agents import *


def main():
    env = ConnectFourGymEnv()

    input_shape = env.observation_space.flatten().shape
    num_actions = len(env.action_space)
    episodes = 5000
    batch_size = 100

    agent = DQNAgent(0, input_shape, num_actions, episodes, model_name="agentv2.hdf5")

    start_state = env.reset()
    start_state = np.reshape(start_state, (1,-1))
    normalized_start_state = agent.normalize_board(start_state)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1,-1))
        
        previous_state = None
        action = None
        
        total_rewards = 0
        gained_reward_on_turn = 0
        
        while not env.game_over:
            normalized_state = agent.normalize_board(state)
            previous_action = action
            
            if type(previous_state) == np.ndarray and env.current_player == agent.player_number:
                agent.memorize(previous_state, previous_action, gained_reward_on_turn, normalized_state, done)
                gained_reward_on_turn = 0
            
            action = agent.make_move(env.agent)
            new_state, reward, done, info = env.step(action)
            
            if agent.player_number == 1:
                reward = reward * -1
            
            previous_state = normalized_state.copy()
            state = np.reshape(new_state, (1,-1))
            
            gained_reward_on_turn += reward
            total_rewards += reward
        
        if env.current_player == agent.player_number:
            normalized_state = agent.normalize_board(state)
            agent.memorize(normalized_state, action, gained_reward_on_turn, normalized_start_state, done)
        else:
            agent.memorize(previous_state, previous_action, gained_reward_on_turn, normalized_state, done)
        
        agent.learn(batch_size)
        
        if random.uniform(0, 1) < 0.5:
            agent.switch_side()
        
        if e % 100 == 0:
            print("episode: {}/{}, epsilon: {:.2f}, average: {:.2f}".format(e, episodes, agent.epsilon, avg_reward))
            agent.save("agentv2.hdf5")


if __name__ == "__main__":
    main()
