import pygame


class GridGame:
    def __init__(self,
                 width: int = 10,
                 height: int = 10,
                 obstacles: list = [],
                 playerxy: tuple = (0, 0),
                 goal: tuple = None,
                 moves: int = None):

        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        for x, y in obstacles:
            self.grid[y][x] = 1

        self.playerxy = playerxy
        self.goal = goal or (width - 1, height - 1)

        self.moves = 0
        self.maxmoves = moves or width + height

        self.grid[playerxy[1]][playerxy[0]] = 2
        self.grid[self.goal[1]][self.goal[0]] = 3

        self.width, self.height = width, height

    def move(self, move:int):
        """move: 0 = up, 1 = down, 2 = left, 3 = right"""
        px, py = self.playerxy
        newplayerxy = px, py

        if move == 0 and py > 0:
            if self.grid[py - 1][px] != 1:
                newplayerxy = px, py - 1
        if move == 1 and py < self.height - 1:
            if self.grid[py + 1][px] != 1:
                newplayerxy = px, py + 1
        if move == 2 and px > 0:
            if self.grid[py][px - 1] != 1:
                newplayerxy = px - 1, py
        if move == 3 and px < self.width - 1:
            if self.grid[py][px + 1] != 1:
                newplayerxy = px + 1, py

        self.grid[py][px] = 0
        px, py = newplayerxy
        self.playerxy = px, py
        self.grid[py][px] = 2
        self.moves += 1

    def init_pygame(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.run = True
        self.scale = 50
        self.screen = pygame.display.set_mode((self.width * self.scale, self.height * self.scale))

    def draw(self):
        colors = {0: (0, 0, 0),
                  1: (255, 255, 255),
                  2: (255, 0, 0),
                  3: (0, 255, 0)}
        for y in range(self.height):
            for x in range(self.width):
                a = self.grid[y][x]
                rx, ry = x * self.scale, y * self.scale
                r = pygame.Rect(rx, ry, self.scale, self.scale)
                pygame.draw.rect(self.screen, colors[a], r)
        pygame.display.flip()

    def actions(self):
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0
                if event.key == pygame.K_s:
                    action = 1
                if event.key == pygame.K_a:
                    action = 2
                if event.key == pygame.K_d:
                    action = 3
        return action

    def win(self):
        if self.playerxy == self.goal:
            return True
        return False

    def play(self):
        self.init_pygame()
        while self.run and self.moves < self.maxmoves:
            self.draw()
            action = self.actions()
            if action is not None:
                self.move(action)
            if self.win():
                self.run = False
            self.clock.tick(30)
        self.stop()

    def stop(self):
        pygame.quit()
