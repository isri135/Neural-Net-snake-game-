"""
Snake game with ML-friendly pure step(state, action) function.
Run headless (no rendering) for thousands of games, or with pygame for visualization.
"""

from dataclasses import dataclass
from typing import Tuple
import random

# Actions: 0=left, 1=straight, 2=right (relative to current direction)
ACTION_LEFT = 0
ACTION_STRAIGHT = 1
ACTION_RIGHT = 2

# Directions: 0=up, 1=right, 2=down, 3=left (clockwise)
DIR_DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (dy, dx)


@dataclass
class SnakeState:
    """Immutable game state."""
    snake: Tuple[Tuple[int, int], ...]  # head at index 0
    direction: int  # 0=up, 1=right, 2=down, 3=left
    food: Tuple[int, int]
    alive: bool
    score: int
    grid_width: int
    grid_height: int

    def copy(self, **kwargs) -> "SnakeState":
        d = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        d.update(kwargs)
        return SnakeState(**d)


def initial_state(grid_width: int = 20, grid_height: int = 20, seed: int | None = None) -> SnakeState:
    """Create initial game state."""
    rng = random.Random(seed)
    cy, cx = grid_height // 2, grid_width // 2
    snake = ((cy, cx),)
    food = _spawn_food(snake, grid_width, grid_height, rng)
    return SnakeState(
        snake=snake,
        direction=0,  # start moving up
        food=food,
        alive=True,
        score=0,
        grid_width=grid_width,
        grid_height=grid_height,
    )


def _spawn_food(
    snake: Tuple[Tuple[int, int], ...],
    grid_width: int,
    grid_height: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """Spawn food at a random empty cell."""
    occupied = set(snake)
    while True:
        y = rng.randint(0, grid_height - 1)
        x = rng.randint(0, grid_width - 1)
        if (y, x) not in occupied:
            return (y, x)


def step(state: SnakeState, action: int, rng: random.Random | None = None) -> SnakeState:
    """
    Pure transition: next_state = step(state, action).
    Use this to run thousands of games without rendering.
    """
    if not state.alive:
        return state.copy()

    rng = rng or random.Random()

    # 1. New direction: left (-1), straight (0), right (+1)
    turn = action - 1  # 0->-1, 1->0, 2->+1
    direction = (state.direction + turn) % 4

    # 2. New head
    dy, dx = DIR_DELTA[direction]
    head = state.snake[0]
    new_head = (head[0] + dy, head[1] + dx)

    # 3. Wall collision
    if not (0 <= new_head[0] < state.grid_height and 0 <= new_head[1] < state.grid_width):
        return state.copy(alive=False, direction=direction)

    # 4. Self collision
    if new_head in state.snake:
        return state.copy(alive=False, direction=direction)

    # 5. Food eaten?
    ate_food = new_head == state.food

    if ate_food:
        new_snake = (new_head,) + state.snake
        new_food = _spawn_food(new_snake, state.grid_width, state.grid_height, rng)
        return state.copy(
            snake=new_snake,
            direction=direction,
            food=new_food,
            score=state.score + 1,
        )
    else:
        new_snake = (new_head,) + state.snake[:-1]
        return state.copy(
            snake=new_snake,
            direction=direction,
        )


# --- Pygame visualization ---

def run_game(
    grid_width: int = 20,
    grid_height: int = 20,
    cell_size: int = 24,
    fps: int = 10,
):
    """Run interactive snake game with pygame (blocking)."""
    import pygame

    pygame.init()
    W = grid_width * cell_size
    H = grid_height * cell_size
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    state = initial_state(grid_width, grid_height)
    rng = random.Random()
    current_action = ACTION_STRAIGHT

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Up/down/left/right -> set direction; 180 turns ignored (would insta-die)
                key_to_dir = {
                    pygame.K_UP: 0,
                    pygame.K_RIGHT: 1,
                    pygame.K_DOWN: 2,
                    pygame.K_LEFT: 3,
                }
                if event.key in key_to_dir:
                    want = key_to_dir[event.key]
                    d = state.direction
                    if want == d:
                        current_action = ACTION_STRAIGHT
                    elif want == (d - 1) % 4:
                        current_action = ACTION_LEFT
                    elif want == (d + 1) % 4:
                        current_action = ACTION_RIGHT
                    # else: 180 turn, ignore

        if state.alive:
            state = step(state, current_action, rng)
            current_action = ACTION_STRAIGHT  # reset after each step

        # Draw
        screen.fill((20, 20, 30))
        for i, (y, x) in enumerate(state.snake):
            c = (100, 200, 100) if i == 0 else (60, 160, 60)
            rect = (x * cell_size, y * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, c, rect)

        fy, fx = state.food
        pygame.draw.rect(
            screen, (220, 80, 80),
            (fx * cell_size, fy * cell_size, cell_size - 1, cell_size - 1),
        )

        pygame.display.flip()
        clock.tick(fps)

        if not state.alive:
            pygame.time.wait(1500)
            running = False

    pygame.quit()
    return state.score


def run_headless_example(n_games: int = 1000):
    """Example: run many games without rendering (for ML)."""
    total_score = 0
    rng = random.Random(42)
    for _ in range(n_games):
        state = initial_state(20, 20, seed=rng.randint(0, 2**31 - 1))
        while state.alive:
            action = rng.randint(0, 2)  # random policy
            state = step(state, action, rng)
        total_score += state.score
    print(f"Ran {n_games} games (random policy), avg score: {total_score / n_games:.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        run_headless_example(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    else:
        run_game()
