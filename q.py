import textwrap
import time

import numpy as np
import pygame
import pygame.freetype
from pygame.locals import KEYDOWN, K_q


VACCUM = -10
BONE = 100

class Map:
    VACCUM = 'v'

    def __init__(self):
        with open('map') as f:
            self.map = list(f.read())

        self.COLS = self.map.index('\n')
        self.ROWS = self.map.count('\n')

        # Filter newline chars after getting row and column count
        self.map = [m for m in self.map if m != '\n']

        # Must set these values after removing newline chars
        player_symbol_loc = self.map.index('e')
        self.GOAL = self.map.index('b')

        self.player_loc = self.index_to_coords(player_symbol_loc)

        # Store initial values to be able to reset
        self._init_map = self.map.copy()
        self._init_loc = self.player_loc

    def player_loc_to_map_index(self, x=None, y=None):
        if x is None:
            x = self.player_loc[0]

        if y is None:
            y = self.player_loc[1]

        return self.ROWS * x + y

    def index_to_coords(self, i):
        x = i // self.ROWS
        return x, i - (self.ROWS * x)

    def whats_there(self, x, y):
        return self.map[self.player_loc_to_map_index(x, y)]

    def update_map(self):
        self.map[self.map.index('e')] = '.'
        self.map[self.player_loc_to_map_index()] = 'e'

    def _can_move(self, x_offset=0, y_offset=0):
        x, y = self.player_loc
        x += x_offset
        y += y_offset

        if not 0 <= x < self.ROWS or not 0 <= y < self.COLS or self.whats_there(x, y) == self.VACCUM:
            return False

        return True

    def can_move_up(self):
        return self._can_move(x_offset=-1)

    def can_move_down(self):
        return self._can_move(x_offset=1)

    def can_move_left(self):
        return self._can_move(y_offset=-1)

    def can_move_right(self):
        return self._can_move(y_offset=1)

    def _move(self, x_offset=0, y_offset=0):
        x, y = self.player_loc
        self.player_loc = x + x_offset, y + y_offset
        self.update_map()

    def move_up(self):
        if self._can_move(x_offset=-1):
            self._move(x_offset=-1)

    def move_down(self):
        if self._can_move(x_offset=1):
            self._move(x_offset=1)

    def move_left(self):
        if self._can_move(y_offset=-1):
            self._move(y_offset=-1)

    def move_right(self):
        if self._can_move(y_offset=1):
            self._move(y_offset=1)

    def reset(self):
        self.map = self._init_map
        self.player_loc = self._init_loc

    def done(self):
        return self.GOAL == self.player_loc_to_map_index()

    def __repr__(self):
        return '\n'.join(textwrap.wrap(''.join(self.map), self.ROWS))


class Agent:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    STEP_PENALTY = -1
    INVALID_PENALTY = -100
    GOAL = 1_000

    def __init__(self, alpha=0.2, gamma=0.7, epsilon=0.075, map=Map()):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.map = map

        # Q table which is a 3d array of rows * columns * number of actions
        self.q = np.zeros((self.map.ROWS, self.map.COLS, 4))

        # Set of possible actions mapping to validation of the action, execution of the action, and x and y offsets
        self.ACTIONS = {
            self.UP: {
                'valid': self.map.can_move_up,
                'move': self.map.move_up,
                'x_offset': -1,
                'y_offset': 0
            },
            self.DOWN: {
                'valid': self.map.can_move_down,
                'move': self.map.move_down,
                'x_offset': 1,
                'y_offset': 0
            },
            self.LEFT: {
                'valid': self.map.can_move_left,
                'move': self.map.move_left,
                'x_offset': 0,
                'y_offset': -1
            },
            self.RIGHT: {
                'valid': self.map.can_move_right,
                'move': self.map.move_right,
                'x_offset': 0,
                'y_offset': 1
            }
        }

    def step(self, action):
        next_x, next_y = self.map.player_loc

        action = self.ACTIONS[action]
        if action['valid']():
            next_x += action['x_offset']
            next_y += action['y_offset']

            if self.map.index_to_coords(self.map.GOAL) == (next_x, next_y):
                reward = self.GOAL
            else:
                reward = self.STEP_PENALTY

        else:
            reward = self.INVALID_PENALTY

        return next_x, next_y, reward

    def learn(self, episodes=10_000):
        np.set_printoptions(
            precision=4, suppress=True,
            linewidth=150
        )
        step_tracker = []

        for i in range(episodes):
            self.map.reset()

            steps, penalties = 0, 0
            while not self.map.done() and steps < self.map.ROWS ** 2 * self.map.COLS ** 2:
                agent_x, agent_y = self.map.player_loc

                # Explore or exploit
                if np.random.uniform() < self.epsilon:
                    action = np.random.choice(list(self.ACTIONS.keys()))
                    action_str = 'explr'
                else:
                    action = np.argmax(self.q[self.map.player_loc])
                    action_str = 'explt'

                next_x, next_y, reward = self.step(action)
                next_max = self.q[next_x, next_y].max()

                new_q_value = (1 - self.alpha) * self.q[agent_x, agent_y, action] + self.alpha * (reward + self.gamma * next_max)
                # print(f'episode={i} loc={self.map.player_loc}, action={action}({action_str}) next_x={next_x} next_y={next_y} reward={reward}, next_max={round(next_max, 4)} steps={steps} pen={penalties}')
                # print(self.q[agent_x, agent_y])

                if reward == self.INVALID_PENALTY:
                    penalties += 1

                self.q[agent_x, agent_y, action] = new_q_value
                self.ACTIONS[action]['move']()
                steps += 1

                q_max_str = np.around(np.amax(self.q, axis=2), decimals=4).astype('str')
                map_array = np.array(self.map.map).reshape(q_max_str.shape)
                yield np.where(map_array == '.', q_max_str, map_array)

            step_tracker.append({'steps': steps, 'penalties': penalties, 'finished': self.map.done()})
            # print(self.map)
            # print(np.amax(self.q, axis=2))
            # print(i, step_tracker[i])

            last_10 = step_tracker[-10:]
            if i > 10 and all([last_10[0] == l for l in last_10]):
                print('likely optimised... finished learning')
                break

        print()
        print('\n\nbest performance')
        best_steps = sorted([{'episode': i, **s} for i, s in enumerate(step_tracker)], key=lambda s: s['steps'])
        print('\n'.join(map(str, best_steps[:5])))


class Animate:
    BOX_DIM = 79.5

    BLACK = 0, 0, 0
    GREY = 160, 160, 160
    LINE_WIDTH = 2

    def __init__(self, agent=Agent()):
        self.agent = agent
        self._dimensions = 1_200, 800

        self._grid_dim = self.BOX_DIM * self.agent.map.ROWS
        self._grid_x, self._grid_y = (self._dimensions[0] - self._grid_dim) / 2, (self._dimensions[1] - self._grid_dim) / 2

    def animate(self):
        pygame.init()

        self.display = pygame.display.set_mode(self._dimensions)
        self._font = pygame.freetype.SysFont('monospace', 12)

        # Load and scale images
        self._dog = pygame.transform.scale(pygame.image.load('./img/dog.jpeg'), list(map(int, (self.BOX_DIM, self.BOX_DIM))))
        self._shower = pygame.transform.scale(pygame.image.load('./img/shower.png'), list(map(int, (self.BOX_DIM, self.BOX_DIM))))
        self._steak = pygame.transform.scale(pygame.image.load('./img/steak.jpeg'), list(map(int, (self.BOX_DIM, self.BOX_DIM))))

        playing = True
        learning = self.agent.learn()
        while playing:
            self.display.fill(self.GREY)
            self._draw_grid()
            self._populate_grid(next(learning, None))
            pygame.display.update()
            playing = self._check_events()

    def _draw_line(self, start, end):
        pygame.draw.line(self.display, self.BLACK, start, end, self.LINE_WIDTH)

    def _draw_grid(self):
        # Draw borders
        # Top
        self._draw_line(
            (self._grid_x, self._grid_y),
            (self._grid_x + self._grid_dim, self._grid_y)
        )
        # Bottom
        self._draw_line(
            (self._grid_x, self._grid_y + self._grid_dim),
            (self._grid_x + self._grid_dim, self._grid_y + self._grid_dim)
        )
        # Left
        self._draw_line(
            (self._grid_x, self._grid_y),
            (self._grid_x, self._grid_y + self._grid_dim)
        )
        # Right
        self._draw_line(
            (self._grid_x + self._grid_dim, self._grid_y),
            (self._grid_x + self._grid_dim, self._grid_y + self._grid_dim)
        )

        # Draw grid horizontal lines
        for i in range(self.agent.map.ROWS):
            y = self._grid_y + i * self.BOX_DIM
            self._draw_line((self._grid_x, y), (self._grid_x + self._grid_dim, y))

        # Draw grid vertical lines
        for i in range(self.agent.map.COLS):
            x = self._grid_x + i * self.BOX_DIM
            self._draw_line((x, self._grid_y), (x, self._grid_y + self._grid_dim))

    def _populate_grid(self, grid):
        if grid is None:
            return

        # Sometimes we lose the goal
        grid[self.agent.map.index_to_coords(self.agent.map.GOAL)] = 'b'

        rows, cols = grid.shape
        for i in range(rows):
            for j in range(cols):
                text = grid[i, j]
                x, y = self._grid_x + j * self.BOX_DIM, self._grid_y + i * self.BOX_DIM

                if text == 'e':
                    self.display.blit(self._dog, (x, y))

                elif text == 'b':
                    self.display.blit(self._steak, (x, y))

                elif text == 'v':
                    self.display.blit(self._shower, (x, y))

                else:
                    score = self._font.get_rect(text)
                    score.center = x + 0.5 * self.BOX_DIM, y + 0.5 * self.BOX_DIM
                    self._font.render_to(self.display, score.topleft, text, self.BLACK)

        pygame.display.flip()

    def _check_events(self):
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT:
                return False
            elif event.type == KEYDOWN and event.key == K_q:
                pygame.quit()
                return False

        return True

def main():
    # a = Agent()
    # learn = a.learn(episodes=10_000)
    # for i in range(100):
    #     print(next(learn))

    animate = Animate()
    animate.animate()


if __name__ == '__main__':
    main()
