import unittest

import torch

from src.model import WavefunctionPINN
from src.physics import exact_ground_state, exact_state, schrodinger_residual


class PhysicsTests(unittest.TestCase):
    def test_exact_ground_state_is_normalized_on_finite_domain(self):
        x = torch.linspace(-8, 8, 4001).reshape(-1, 1)
        dx = x[1] - x[0]
        integral = torch.sum(exact_ground_state(x) ** 2) * dx
        self.assertAlmostEqual(float(integral), 1.0, places=3)

    def test_exact_ground_state_satisfies_equation(self):
        x = torch.linspace(-4, 4, 101).reshape(-1, 1).requires_grad_()
        model = lambda values: exact_ground_state(values)
        residual = schrodinger_residual(model, x, torch.tensor(0.5))
        self.assertLess(float(torch.max(torch.abs(residual)).detach()), 1e-5)

    def test_model_output_shape(self):
        model = WavefunctionPINN(width=16, depth=2)
        self.assertEqual(tuple(model(torch.zeros(5, 1)).shape), (5, 1))

    def test_first_two_exact_states_are_orthogonal(self):
        x = torch.linspace(-8, 8, 4001).reshape(-1, 1)
        overlap = torch.trapezoid((exact_state(x, 0) * exact_state(x, 1))[:, 0], x[:, 0])
        self.assertAlmostEqual(float(overlap), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
