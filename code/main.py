from connectFour import *
from connectFourGame import *

def main():
    player1 = 'random'
    player2 = 'random'
    
    connect_four_game = connectFourAgent()
    
    game = ConnectFourGameAgent(connect_four_game, player1, player2)
    game.play()
    

if __name__ == "__main__":
    main()
