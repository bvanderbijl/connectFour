import pygame
import sys
import math

from connect4 import *
from agents import *

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)


class ConnectFourGameAgent:
    def __init__(self, agent, player1="human", player2="human"):
        self.agent = agent
        self.width = COLUMN_COUNT * SQUARESIZE
        self.height = (ROW_COUNT+1) * SQUARESIZE
        self.size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.size)
        
        self.player_agents = {0: self.select_user(player1, 1), 1: self.select_user(player2, 2)}
        
        pygame.init()
        self.font = pygame.font.SysFont("monospace", 75)
        
        
    def draw_board(self):
        board = self.agent.board
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == 0:
                    pygame.draw.circle(self.screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), self.height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif board[r][c] == 1: 
                    pygame.draw.circle(self.screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), self.height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()
    
    def play(self):
        self.draw_board()
        
        while not self.agent.game_finished():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                    
                if self.player_agents[self.agent.current_player] == None:
                    if event.type == pygame.MOUSEMOTION:
                        pygame.draw.rect(self.screen, BLACK, (0,0, self.width, SQUARESIZE))
                        posx = event.pos[0]
                        if self.agent.current_player == 0:
                            pygame.draw.circle(self.screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                        else: 
                            pygame.draw.circle(self.screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
                    pygame.display.update()
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pygame.draw.rect(self.screen, BLACK, (0,0, self.width, SQUARESIZE))
                        posx = event.pos[0]
                        col = int(math.floor(posx/SQUARESIZE))
                        
                        if self.agent.is_valid_move(col):
                            self.agent.drop_piece(col+1)
                            
                        self.draw_board()
                else:
                    pygame.time.wait(1000)
                    self.agent.drop_piece(self.player_agents[self.agent.current_player].make_move(self.agent))
                    
                    self.draw_board()
                
            if self.agent.game_finished():
                winner = self.agent.next_move() + 1
                color = RED if winner == 1 else YELLOW
                label = self.font.render(f"Player {winner} wins!!", 1, color)
                self.screen.blit(label, (40,10))
                pygame.display.update()
                
                pygame.time.wait(3000)
    
    def select_user(self, user_type, player_number):
        if user_type == 'human':
            return None
        elif user_type == 'random':
            return RandomConnectPlayer(player_number)
        else:
            raise UnknownUserException()

class UnknownUserException():
    pass