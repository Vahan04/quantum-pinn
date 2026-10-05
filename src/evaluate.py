import torch

from .physics import exact_state


def relative_l2_error(prediction, target):
    return torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target)


def evaluate_model(model, x, n=0):
    with torch.no_grad():
        prediction = model(x)
        target = exact_state(x, n)
        sign = torch.sign(torch.sum(prediction * target))
        prediction = prediction * sign
        return {
            "relative_l2_error": float(relative_l2_error(prediction, target)),
            "max_absolute_error": float(torch.max(torch.abs(prediction - target))),
            "normalization_error": float(torch.abs(torch.trapezoid(prediction[:, 0] ** 2, x[:, 0]) - 1.0)),
        }
