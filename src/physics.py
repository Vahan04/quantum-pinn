import torch


def exact_ground_state(x):
    """Return the normalized ground state in dimensionless units."""
    return exact_state(x, 0)


def exact_state(x, n):
    """Return the normalized nth Hermite-Gaussian state."""
    h0 = torch.ones_like(x)
    if n == 0:
        polynomial = h0
    else:
        h1 = 2 * x
        if n == 1:
            polynomial = h1
        else:
            for order in range(1, n):
                h0, h1 = h1, 2 * x * h1 - 2 * order * h0
            polynomial = h1
    normalization = 1.0 / torch.sqrt((2.0**n) * torch.tensor(float(factorial(n)), device=x.device, dtype=x.dtype))
    return torch.pi ** (-0.25) * normalization * polynomial * torch.exp(-0.5 * x**2)


def factorial(n):
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


def second_derivative(y, x):
    first = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return torch.autograd.grad(
        first,
        x,
        grad_outputs=torch.ones_like(first),
        create_graph=True,
        retain_graph=True,
    )[0]


def schrodinger_residual(model, x, energy):
    """Compute Hψ - Eψ for H = -1/2 d²/dx² + x²/2."""
    psi = model(x)
    d2psi = second_derivative(psi, x)
    potential = 0.5 * x**2
    return -0.5 * d2psi + potential * psi - energy * psi


def normalization_loss(psi, dx):
    integral = torch.sum(psi**2) * dx
    return (integral - 1.0) ** 2


def boundary_loss(model, left, right):
    return model(left).pow(2).mean() + model(right).pow(2).mean()
