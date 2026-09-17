"""Analytic zero-Dirichlet test pairs for the Poisson experiments.

Each pair satisfies f = u_xx + u_yy on [-1, 1]^2.  The functions accept
PyTorch tensors and can be assigned directly to ``u`` and ``f`` in the
evaluation part of ``main2.ipynb``.
"""

import torch


def u_multimode(x, y):
    """Unseen mixed-frequency Fourier solution (contains modes above K=4)."""
    return (
        torch.sin(torch.pi * (x + 1) / 2)
        * torch.sin(2 * torch.pi * (y + 1) / 2)
        + 0.35
        * torch.sin(5 * torch.pi * (x + 1) / 2)
        * torch.sin(3 * torch.pi * (y + 1) / 2)
        - 0.20
        * torch.sin(7 * torch.pi * (x + 1) / 2)
        * torch.sin(torch.pi * (y + 1) / 2)
    )


def f_multimode(x, y):
    """Analytic Laplacian of :func:`u_multimode`."""
    pi2_over_4 = torch.pi**2 / 4
    return (
        -5 * pi2_over_4
        * torch.sin(torch.pi * (x + 1) / 2)
        * torch.sin(2 * torch.pi * (y + 1) / 2)
        - 0.35 * 34 * pi2_over_4
        * torch.sin(5 * torch.pi * (x + 1) / 2)
        * torch.sin(3 * torch.pi * (y + 1) / 2)
        + 0.20 * 50 * pi2_over_4
        * torch.sin(7 * torch.pi * (x + 1) / 2)
        * torch.sin(torch.pi * (y + 1) / 2)
    )


def u_polynomial(x, y):
    """Smooth non-Fourier polynomial solution with exact zero boundary."""
    return (1 - x**2) * (1 - y**2) * (1 + 0.5 * x + 0.25 * y)


def f_polynomial(x, y):
    """Analytic Laplacian of :func:`u_polynomial`."""
    c = 1 + 0.5 * x + 0.25 * y
    return (1 - y**2) * (-2 * c - 2 * x) + (1 - x**2) * (-2 * c - y)


TEST_CASES = {
    "multimode": (u_multimode, f_multimode),
    "polynomial": (u_polynomial, f_polynomial),
}
