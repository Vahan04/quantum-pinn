# Quantum PINN: Ground State of the Harmonic Oscillator

This project uses Physics-Informed Neural Networks (PINNs) to learn the first four eigenstates of the one-dimensional quantum harmonic oscillator.

The time-independent Schrödinger equation is

```text
[-1/2 d²/dx² + 1/2 x²] ψ(x) = E ψ(x)
```

in dimensionless units where `ℏ = m = ω = 1`. The exact ground state is

```text
ψ₀(x) = π^(-1/4) exp(-x² / 2),    E₀ = 1/2
```

The network is trained with four physics-aware objectives:

1. Schrödinger residual at collocation points.
2. Boundary condition `ψ(-L) = ψ(L) = 0`.
3. Normalization `∫ |ψ|² dx = 1`.
4. Orthogonality to states already learned.
5. Even/odd parity constraints for each state.

## Run

Create an environment and install the small dependency set:

```bash
cd quantum-pinn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --epochs 1500 --states 4
```

The script writes `results/spectrum.png` and `results/metrics.json`.

Run tests with:

```bash
python -m unittest discover -s tests
```

## Project structure

```text
quantum-pinn/
├── src/
│   ├── physics.py       # equation, eigenstates, and losses
│   ├── model.py         # PINN architecture
│   ├── train.py         # training and evaluation entry point
│   └── evaluate.py      # reusable error metrics
├── tests/
├── results/
├── requirements.txt
└── README.md
```
