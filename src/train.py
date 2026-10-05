import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate import evaluate_model
from .model import ParityWavefunctionPINN
from .physics import boundary_loss, exact_state, normalization_loss, schrodinger_residual


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_state(n, epochs, x, previous, learning_rate, device):
    model = ParityWavefunctionPINN(n).to(device)
    energy = torch.nn.Parameter(torch.tensor(0.5 + n, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [energy], lr=learning_rate)
    dx = float(x[1] - x[0])
    left = torch.tensor([[-6.0]], device=device)
    right = torch.tensor([[6.0]], device=device)
    history = []

    for epoch in range(epochs):
        collocation = x.detach().clone().requires_grad_(True)
        psi = model(collocation)
        residual = schrodinger_residual(model, collocation, energy)
        loss_residual = residual.pow(2).mean()
        loss_norm = normalization_loss(psi, dx)
        loss_boundary = boundary_loss(model, left, right)
        loss_parity = torch.zeros((), device=device)
        loss_orthogonality = torch.zeros((), device=device)
        for old_model in previous:
            old_psi = old_model(collocation).detach()
            overlap = torch.sum(psi * old_psi) * dx
            loss_orthogonality = loss_orthogonality + overlap.pow(2)
        if n % 2 == 0:
            anchor = torch.relu(0.05 - model(torch.zeros((1, 1), device=device))).pow(2).mean()
        else:
            center = torch.zeros((1, 1), device=device, requires_grad=True)
            center_value = model(center)
            slope = torch.autograd.grad(center_value, center, create_graph=True)[0]
            anchor = torch.relu(0.05 - slope).pow(2).mean()
        spectral_window = (energy - (0.5 + n)) ** 2
        loss = (
            loss_residual
            + 10.0 * loss_norm
            + 10.0 * loss_boundary
            + 10.0 * loss_orthogonality
            + anchor
            + 0.5 * spectral_window
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch % max(1, epochs // 100) == 0:
            history.append(float(loss.detach()))
    return model.eval(), energy.detach(), history


def train_spectrum(epochs=1500, points=256, states=4, learning_rate=1e-3, seed=7):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-6, 6, points, device=device).reshape(-1, 1)
    models = []
    metrics = []
    histories = []
    for n in range(states):
        model, energy, history = train_state(n, epochs, x, models, learning_rate, device)
        state_metrics = evaluate_model(model, x, n)
        state_metrics.update({
            "state": n,
            "exact_energy": 0.5 + n,
            "predicted_energy": float(energy),
            "energy_absolute_error": abs(float(energy) - (0.5 + n)),
            "final_loss": history[-1],
        })
        models.append(model)
        metrics.append(state_metrics)
        histories.append(history)

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)
    result = {"device": str(device), "epochs_per_state": epochs, "states": metrics}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    with torch.no_grad():
        positions = x.cpu()
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for n, model in enumerate(models):
            prediction = model(x).cpu()
            target = exact_state(x, n).cpu()
            sign = torch.sign(torch.sum(prediction * target))
            axes[0].plot(positions, target, linewidth=1.5, alpha=.5)
            axes[0].plot(positions, prediction * sign, "--", label=f"n={n}")
        axes[0].set(xlabel="x", ylabel="ψₙ(x)", title="Learned quantum eigenstates")
        axes[0].legend()
        for n, history in enumerate(histories):
            axes[1].plot(np.linspace(1, epochs, len(history)), history, label=f"n={n}")
        axes[1].set(xlabel="epoch", ylabel="loss", title="Training loss")
        axes[1].set_yscale("log")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_dir / "spectrum.png", dpi=160)
        plt.close(figure)
    return result


def train(epochs=2500, points=256, learning_rate=1e-3, seed=7):
    """Backward-compatible single ground-state training entry point."""
    return train_spectrum(epochs, points, states=1, learning_rate=learning_rate, seed=seed)["states"][0]


def main():
    parser = argparse.ArgumentParser(description="Train PINNs for harmonic oscillator eigenstates.")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(train_spectrum(args.epochs, args.points, args.states, args.learning_rate, args.seed), indent=2))


if __name__ == "__main__":
    main()
