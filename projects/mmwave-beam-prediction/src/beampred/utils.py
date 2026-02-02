import numpy as np
from beampred.config import N_ANTENNAS, D_SPACING, WAVELENGTH


def array_response_vector(theta, n_antennas=N_ANTENNAS):
    n = np.arange(n_antennas)
    phase = 2 * np.pi * D_SPACING / WAVELENGTH * n * np.sin(theta)
    return np.exp(1j * phase) / np.sqrt(n_antennas)


def to_db(x, floor=-100):
    return np.maximum(10 * np.log10(np.maximum(x, 1e-30)), floor)


def from_db(x_db):
    return 10 ** (x_db / 10)


def path_loss_umi(d, los=True):
    d = np.maximum(d, 1.0)
    if los:
        pl = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(28.0)
    else:
        pl = 32.4 + 31.9 * np.log10(d) + 20.0 * np.log10(28.0)
    return pl
