"""Policy-value network: AlphaZero's residual CNN, scaled down (proposal: 4-6 blocks, 64-128 filters).

Input:  NUM_PLANES x 8 x 8 planes from `encoding.encode_state`, from the side to move's view.
Output: policy logits over the 4,096 canonical from/to actions (illegal ones are masked by the
        caller), and a value in [-1, 1]: the expected result for the side to move.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ACTION_SIZE, NUM_PLANES, encode_state, move_to_action
from .game import GameState


@dataclass(frozen=True)
class NetConfig:
    blocks: int = 6          # residual blocks
    channels: int = 64       # filters per convolution
    policy_channels: int = 2  # 1x1 convolution before the policy's fully connected layer
    value_hidden: int = 64   # hidden units in the value head


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, config: NetConfig = NetConfig()):
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*(ResidualBlock(c) for _ in range(config.blocks)))
        p = config.policy_channels
        self.policy_conv = nn.Sequential(nn.Conv2d(c, p, 1, bias=False), nn.BatchNorm2d(p), nn.ReLU())
        self.policy_fc = nn.Linear(p * 64, ACTION_SIZE)
        self.value_conv = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(64, config.value_hidden), nn.ReLU(), nn.Linear(config.value_hidden, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, NUM_PLANES, 8, 8) -> (policy logits (B, 4096), value (B,) in [-1, 1])."""
        h = self.body(self.stem(x))
        logits = self.policy_fc(self.policy_conv(h).flatten(1))
        value = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1)
        return logits, value


def new_network(config: NetConfig = NetConfig(), seed: int = 0) -> PolicyValueNet:
    """A freshly initialised network. The same seed always gives the same weights."""
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    net = PolicyValueNet(config)
    torch.random.set_rng_state(generator_state)  # do not disturb the caller's random stream
    return net.eval()


def parameter_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def default_device() -> torch.device:
    """CUDA if present (Colab), else the CPU. Apple's MPS is slower than the CPU for the single,
    tiny batches MCTS sends, so it is only used when asked for explicitly."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetworkEvaluator:
    """Evaluates positions for MCTS with a policy-value network.

    Returns priors over the legal moves (engine move numbers, softmax of the masked logits) and
    the value for the side to move.
    """

    def __init__(self, net: PolicyValueNet, device: torch.device | None = None):
        self.device = device or default_device()
        self.net = net.to(self.device).eval()

    def evaluate(self, state: GameState) -> tuple[dict[int, float], float]:
        return self.evaluate_many([state])[0]

    @torch.inference_mode()
    def evaluate_many(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        x = torch.from_numpy(np.stack([encode_state(s) for s in states])).to(self.device)
        logits, values = self.net(x)
        logits, values = logits.float().cpu().numpy(), values.float().cpu().numpy()
        out = []
        for state, row, value in zip(states, logits, values):
            moves = state.legal_moves()
            side, setup = state.side, state.rules.setup
            chosen = row[[move_to_action(m, side, setup) for m in moves]]
            chosen = np.exp(chosen - chosen.max())
            chosen /= chosen.sum()
            out.append(({m: float(p) for m, p in zip(moves, chosen)}, float(value)))
        return out


def save_checkpoint(path: str, net: PolicyValueNet, **extra) -> None:
    torch.save({"config": asdict(net.config), "state_dict": net.state_dict(), **extra}, path)


def load_checkpoint(path: str, device: torch.device | None = None) -> tuple[PolicyValueNet, dict]:
    data = torch.load(path, map_location=device or "cpu", weights_only=True)
    net = PolicyValueNet(NetConfig(**data.pop("config")))
    net.load_state_dict(data.pop("state_dict"))
    return net.eval(), data
