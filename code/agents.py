import random


class RandomConnectPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
    
    def make_move(self, agent):
        moves = agent.get_valid_moves()
        return random.choice(moves)
    