# game/tetris.py

import pygame
import random

# Costanti per il gioco
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 30
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

# Definizione delle forme dei tetramini
SHAPES = [
    [[1, 1, 1],
     [0, 1, 0]],  # T

    [[1, 1, 0],
     [0, 1, 1]],  # S

    [[0, 1, 1],
     [1, 1, 0]],  # Z

    [[1, 1, 1, 1]],  # I

    [[1, 1],
     [1, 1]],  # O

    [[1, 0, 0],
     [1, 1, 1]],  # J

    [[0, 0, 1],
     [1, 1, 1]]   # L
]

# Colorazione dei pezzi
SHAPE_COLORS = [
    (128, 0, 128),  # Purple
    (0, 255, 0),    # Green
    (255, 0, 0),    # Red
    (0, 255, 255),  # Cyan
    (255, 255, 0),  # Yellow
    (0, 0, 255),    # Blue
    (255, 165, 0)   # Orange
]


class Tetromino:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
        self.x = GRID_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate_clockwise(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

    def rotate_counterclockwise(self):
        self.shape = [list(row) for row in zip(*self.shape)][::-1]


class Tetris:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris - Gesture Control")
        self.clock = pygame.time.Clock()
        self.grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.game_over = False
        self.paused = False

    def new_piece(self):
        index = random.randint(0, len(SHAPES) - 1)
        return Tetromino(SHAPES[index], SHAPE_COLORS[index])

    def check_collision(self, dx=0, dy=0, shape=None):
        shape = shape or self.current_piece.shape
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    nx = self.current_piece.x + x + dx
                    ny = self.current_piece.y + y + dy
                    if nx < 0 or nx >= GRID_WIDTH or ny >= GRID_HEIGHT:
                        return True
                    if ny >= 0 and self.grid[ny][nx] != BLACK:
                        return True
        return False

    def lock_piece(self):
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color
        self.clear_lines()
        self.current_piece = self.new_piece()
        if self.check_collision():
            self.game_over = True

    def clear_lines(self):
        new_grid = [row for row in self.grid if BLACK in row]
        lines_cleared = GRID_HEIGHT - len(new_grid)
        for _ in range(lines_cleared):
            new_grid.insert(0, [BLACK] * GRID_WIDTH)
        self.grid = new_grid

    def move(self, dx, dy):
        if not self.check_collision(dx, dy):
            self.current_piece.x += dx
            self.current_piece.y += dy

    def rotate(self, clockwise=True):
        original_shape = self.current_piece.shape
        if clockwise:
            self.current_piece.rotate_clockwise()
        else:
            self.current_piece.rotate_counterclockwise()
        if self.check_collision():
            self.current_piece.shape = original_shape

    def drop(self):
        while not self.check_collision(dy=1):
            self.current_piece.y += 1
        self.lock_piece()

    def toggle_pause(self):
        self.paused = not self.paused

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color = self.grid[y][x]
                pygame.draw.rect(self.screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.screen, GRAY, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Disegna il tetramino attuale
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    px = (self.current_piece.x + x) * BLOCK_SIZE
                    py = (self.current_piece.y + y) * BLOCK_SIZE
                    pygame.draw.rect(self.screen, self.current_piece.color, (px, py, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.screen, GRAY, (px, py, BLOCK_SIZE, BLOCK_SIZE), 1)

    def run_frame(self, key_events):
        if self.game_over or self.paused:
            return

        for event in key_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    self.move(-1, 0)
                elif event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.move(1, 0)
                elif event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.move(0, 1)
                elif event.key == ord('q'):
                    self.rotate(clockwise=False)
                elif event.key == ord('e'):
                    self.rotate(clockwise=True)
                elif event.key == ord('p'):
                    self.toggle_pause()

        if not self.check_collision(dy=1):
            self.current_piece.y += 1
        else:
            self.lock_piece()

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_grid()
        pygame.display.flip()

    def tick(self):
        self.clock.tick(4)