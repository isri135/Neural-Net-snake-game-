"""
DQN agent that learns to play Snake via reinforcement learning.
Uses the headless step() from snake_game for fast training.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from snake_game import (
    SnakeState,
    initial_state,
    step,
    ACTION_LEFT,
    ACTION_STRAIGHT,
    ACTION_RIGHT,
    DIR_DELTA,
)

# --- State encoding ---

STATE_SIZE = 10

def state_to_features(state: SnakeState) -> np.ndarray:
    """
    Encode game state as a feature vector for the neural network.
    All features are from the snake's perspective (relative to current direction).
    """
    if not state.alive:
        return np.zeros(STATE_SIZE, dtype=np.float32)  # placeholder

    head = state.snake[0]
    body = set(state.snake[1:])
    food = state.food
    d = state.direction
    w, h = state.grid_width, state.grid_height

    # Directions relative to snake: left, straight, right (indices into DIR_DELTA)
    rel_dirs = [(d - 1) % 4, d, (d + 1) % 4]

    # Danger: wall or body in each of 3 directions (1 = danger, 0 = safe)
    danger = []
    for rd in rel_dirs:
        dy, dx = DIR_DELTA[rd]
        ny, nx = head[0] + dy, head[1] + dx
        hit_wall = not (0 <= ny < h and 0 <= nx < w)
        hit_body = (ny, nx) in body
        danger.append(1.0 if (hit_wall or hit_body) else 0.0)

    # Food direction: is food to our left, straight, or right? (one-hot-ish)
    fy, fx = food
    hy, hx = head
    # Vector from head to food
    to_food_y = fy - hy
    to_food_x = fx - hx

    # Project onto snake's local axes (depends on direction)
    # d=0 (up): forward=-y, left=-x, right=+x
    # d=1 (right): forward=+x, left=-y, right=+y
    # d=2 (down): forward=+y, left=+x, right=-x
    # d=3 (left): forward=-x, left=+y, right=-y
    if d == 0:   # up
        fwd, left, right = -to_food_y, -to_food_x, to_food_x
    elif d == 1:  # right
        fwd, left, right = to_food_x, -to_food_y, to_food_y
    elif d == 2:  # down
        fwd, left, right = to_food_y, to_food_x, -to_food_x
    else:         # left
        fwd, left, right = -to_food_x, to_food_y, -to_food_y

    # Normalize by max possible distance
    max_d = max(w, h)
    fwd_n = np.clip(fwd / max_d, -1, 1)
    left_n = np.clip(left / max_d, -1, 1)
    right_n = np.clip(right / max_d, -1, 1)

    # Features: [danger_left, danger_straight, danger_right, food_fwd, food_left, food_right, ...]
    features = np.array([
        danger[0], danger[1], danger[2],
        fwd_n, left_n, right_n,
        # Extra: distance to food (normalized), snake length hint
        np.sqrt(to_food_y**2 + to_food_x**2) / (w + h),
        len(state.snake) / (w * h),
        # Normalized head position (helps with edges)
        head[0] / h, head[1] / w,
    ], dtype=np.float32)

    return features


# --- DQN ---

class DQN(nn.Module):
    def __init__(self, input_size: int = 10, hidden: int = 128, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Replay buffer ---

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- Agent ---

class DQNAgent:
    def __init__(
        self,
        state_size: int = 10,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update: int = 500,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DQN(input_size=state_size).to(self.device)
        self.target_net = DQN(input_size=state_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 2)

        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.policy_net(x)
            return int(q.argmax(dim=1).item())

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# --- Training ---

def train(
    num_episodes: int = 5000,
    grid_size: int = 12,
    max_steps_per_episode: int = 2000,
    save_path: str = "snake_dqn.pt",
    reward_food: float = 10.0,
    reward_death: float = -10.0,
    reward_step: float = -0.01,
    watch: bool = False,
    watch_interval: int = 1,
    watch_fps: int = 12,
    plot: bool = False,
    plot_interval: int = 50,
):
    rng = random.Random(42)
    agent = DQNAgent(state_size=STATE_SIZE)
    scores = []
    scores_window = deque(maxlen=100)
    losses = []

    screen = None
    clock = None
    if watch:
        import pygame
        pygame.init()
        cell_size = 28
        screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        pygame.display.set_caption("Snake - Learning...")
        clock = pygame.time.Clock()

    for ep in range(num_episodes):
        state = initial_state(grid_size, grid_size, seed=rng.randint(0, 2**31 - 1))
        features = state_to_features(state)
        total_reward = 0
        steps = 0
        render_this_episode = watch and (ep % watch_interval == 0)

        while state.alive and steps < max_steps_per_episode:
            action = agent.select_action(features)
            next_state = step(state, action, rng)

            if next_state.alive:
                reward = reward_step
                if next_state.score > state.score:
                    reward += reward_food
            else:
                reward = reward_death

            next_features = state_to_features(next_state)
            agent.buffer.push(features, action, reward, next_features, not next_state.alive)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            agent.decay_epsilon()

            features = next_features
            state = next_state
            total_reward += reward
            steps += 1

            if render_this_episode and screen is not None:
                import pygame
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        watch = False
                        render_this_episode = False
                screen.fill((20, 20, 30))
                for i, (y, x) in enumerate(state.snake):
                    c = (100, 200, 100) if i == 0 else (60, 160, 60)
                    rect = (x * cell_size, y * cell_size, cell_size - 1, cell_size - 1)
                    pygame.draw.rect(screen, c, rect)
                fy, fx = state.food
                pygame.draw.rect(screen, (220, 80, 80), (fx * cell_size, fy * cell_size, cell_size - 1, cell_size - 1))
                font = pygame.font.Font(None, 32)
                txt = font.render(f"Ep {ep + 1} | Score: {state.score} | ε: {agent.epsilon:.2f}", True, (255, 255, 255))
                screen.blit(txt, (5, 5))
                pygame.display.flip()
                clock.tick(watch_fps)

        scores.append(state.score)
        scores_window.append(state.score)

        if plot and (ep + 1) % plot_interval == 0:
            import matplotlib.pyplot as plt
            if ep == 0 or (ep + 1) == plot_interval:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                fig.suptitle("Neural Network Learning", fontsize=12)
            ax1.clear()
            window = min(200, len(scores))
            ax1.plot(scores[-window:], alpha=0.4, color="steelblue", label="Score")
            if len(scores) >= 100:
                rolling = np.convolve(scores, np.ones(100) / 100, mode="valid")
                ax1.plot(range(99, len(scores)), rolling, color="darkblue", linewidth=2, label="Avg (100)")
            ax1.set_ylabel("Score")
            ax1.set_title("Score per episode")
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            if losses:
                ax2.clear()
                loss_window = min(500, len(losses))
                ax2.plot(losses[-loss_window:], alpha=0.5, color="coral", label="Loss")
                if len(losses) >= 50:
                    loss_rolling = np.convolve(losses, np.ones(50) / 50, mode="valid")
                    ax2.plot(range(49, len(losses)), loss_rolling, color="darkred", linewidth=2, label="Avg (50)")
                ax2.set_ylabel("Loss")
                ax2.set_xlabel("Step")
                ax2.set_title("DQN training loss")
                ax2.legend(loc="upper right")
                ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.pause(0.001)

        if (ep + 1) % 100 == 0:
            avg = np.mean(scores_window)
            print(f"Episode {ep + 1}/{num_episodes} | Avg score (last 100): {avg:.1f} | Epsilon: {agent.epsilon:.3f}")

    if screen is not None:
        import pygame
        pygame.quit()

    if plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return agent


# --- Play with trained agent ---

def play_with_agent(model_path: str = "snake_dqn.pt", grid_size: int = 12, fps: int = 8):
    """Watch the trained agent play."""
    import pygame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_size=STATE_SIZE)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0  # greedy

    pygame.init()
    cell_size = 32
    W = grid_size * cell_size
    H = grid_size * cell_size
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Snake - DQN Agent")
    clock = pygame.time.Clock()

    rng = random.Random()
    state = initial_state(grid_size, grid_size)
    features = state_to_features(state)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if state.alive:
            action = agent.select_action(features, greedy=True)
            state = step(state, action, rng)
            features = state_to_features(state)

        # Draw
        screen.fill((20, 20, 30))
        for i, (y, x) in enumerate(state.snake):
            c = (100, 200, 100) if i == 0 else (60, 160, 60)
            rect = (x * cell_size, y * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, c, rect)
        fy, fx = state.food
        pygame.draw.rect(screen, (220, 80, 80), (fx * cell_size, fy * cell_size, cell_size - 1, cell_size - 1))

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {state.score}", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(fps)

        if not state.alive:
            pygame.time.wait(2000)
            state = initial_state(grid_size, grid_size)
            features = state_to_features(state)

    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the DQN agent")
    parser.add_argument("--play", action="store_true", help="Play with trained agent")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--model", type=str, default="snake_dqn.pt")
    parser.add_argument("--watch", action="store_true", help="Show game window during training")
    parser.add_argument("--watch-interval", type=int, default=1, help="Render every N episodes (default 1)")
    parser.add_argument("--watch-fps", type=int, default=12, help="FPS when watching (default 12)")
    parser.add_argument("--plot", action="store_true", help="Show live learning curve (score & loss)")
    parser.add_argument("--plot-interval", type=int, default=50, help="Update plot every N episodes")
    args = parser.parse_args()

    if args.play:
        play_with_agent(args.model)
    else:
        train(
            num_episodes=args.episodes,
            save_path=args.model,
            watch=args.watch,
            watch_interval=args.watch_interval,
            watch_fps=args.watch_fps,
            plot=args.plot,
            plot_interval=args.plot_interval,
        )
