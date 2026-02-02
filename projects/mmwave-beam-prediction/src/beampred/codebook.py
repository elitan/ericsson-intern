import numpy as np
from beampred.config import N_ANTENNAS, N_NARROW_BEAMS, N_WIDE_BEAMS


def generate_dft_codebook(n_antennas, n_beams):
    codebook = np.zeros((n_beams, n_antennas), dtype=complex)
    for i in range(n_beams):
        phase = 2 * np.pi * np.arange(n_antennas) * i / n_beams
        codebook[i] = np.exp(1j * phase) / np.sqrt(n_antennas)
    return codebook


def get_narrow_codebook():
    return generate_dft_codebook(N_ANTENNAS, N_NARROW_BEAMS)


def get_wide_codebook():
    return generate_dft_codebook(N_ANTENNAS, N_WIDE_BEAMS)


def beam_angles(n_beams):
    return np.arcsin(np.linspace(-1 + 1 / n_beams, 1 - 1 / n_beams, n_beams))
