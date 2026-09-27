"""Generate smooth source fields for Darcy samples."""

import numpy as np


def fourier_modes(max_freq=3):
    modes = [
        (m, n)
        for m in range(-max_freq, max_freq + 1)
        for n in range(-max_freq, max_freq + 1)
        if not (m == 0 and n == 0)
    ]
    return modes


def sample_source_field(points, rng, modes, domain_half_width, sigma0=0.6):
    x, y = points[:, 0], points[:, 1]
    L = domain_half_width
    f = np.zeros(points.shape[0])
    for m, n in modes:
        decay = sigma0 / (1.0 + m * m + n * n)
        cc, cs = rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)
        phase = np.pi * (m * x / L + n * y / L)
        f += decay * (cc * np.cos(phase) + cs * np.sin(phase))
    return f


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = np.random.uniform(-1, 1, size=(500, 2))
    modes = fourier_modes(3)
    f = sample_source_field(pts, rng, modes, domain_half_width=1.0)
    print("n_modes:", len(modes), "f range:", f.min(), f.max(), "std:", f.std())
