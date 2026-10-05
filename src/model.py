import torch
from torch import nn


class WavefunctionPINN(nn.Module):
    """MLP that maps position x to a real-valued wavefunction."""

    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ParityWavefunctionPINN(WavefunctionPINN):
    """PINN with a hard even/odd symmetry constraint."""

    def __init__(self, state, width=64, depth=3):
        super().__init__(width, depth)
        self.state = state

    def forward(self, x):
        raw = self.network(x)
        mirrored = self.network(-x)
        parity = 0.5 * (raw + (-1) ** self.state * mirrored)
        boundary = torch.relu(1.0 - (x / 6.0) ** 2)
        return boundary * parity
