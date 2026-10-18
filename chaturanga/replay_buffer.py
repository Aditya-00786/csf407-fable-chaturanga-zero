"""The replay buffer: the most recent self-play positions, which training samples from.

It keeps whole games, newest last, and drops the oldest games once it holds more than
`capacity` positions, so the network trains on recent play by an improving network rather
than on everything since the start. Positions are stored compactly (half-precision planes,
sparse policy targets) and a sampled batch is expanded to dense tensors.
"""

from collections import deque
from typing import Optional

import numpy as np

from .encoding import ACTION_SIZE
from .selfplay import GameRecord, examples


class ReplayBuffer:
    def __init__(self, capacity: int = 60_000):
        self.capacity = capacity
        self._games: deque = deque()   # (planes f16, actions, shares, z), one entry per game
        self._size = 0
        self._flat: Optional[tuple] = None  # concatenated view, rebuilt after changes
        self.games_added = 0

    def __len__(self) -> int:
        return self._size

    @property
    def games(self) -> int:
        return len(self._games)

    def add(self, record: GameRecord) -> int:
        """Add a game's positions (raises ValueError for an invalid record). Returns their number."""
        planes, actions, shares, z = examples(record)
        self._games.append((planes.astype(np.float16), actions, shares, z))
        self._size += len(z)
        self.games_added += 1
        while self._size > self.capacity and len(self._games) > 1:
            self._size -= len(self._games.popleft()[3])
        self._flat = None
        return len(z)

    def _flatten(self) -> tuple:
        if self._flat is None:
            planes = np.concatenate([g[0] for g in self._games])
            actions = [a for g in self._games for a in g[1]]
            shares = [s for g in self._games for s in g[2]]
            z = np.concatenate([g[3] for g in self._games])
            self._flat = (planes, actions, shares, z)
        return self._flat

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A random batch: planes (B, 15, 8, 8) float32, policy targets (B, 4096), values (B,)."""
        if not self._size:
            raise ValueError("the buffer is empty")
        planes, actions, shares, z = self._flatten()
        idx = rng.integers(0, len(z), size=batch_size)
        pi = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
        for row, i in enumerate(idx):
            pi[row, actions[i].astype(np.int64)] = shares[i]
        return planes[idx].astype(np.float32), pi, z[idx]
