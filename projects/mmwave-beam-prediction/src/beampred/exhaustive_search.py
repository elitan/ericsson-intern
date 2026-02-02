import numpy as np
from beampred.codebook import get_narrow_codebook


def exhaustive_search(channel, codebook=None):
    if codebook is None:
        codebook = get_narrow_codebook()
    gains = np.abs(codebook.conj() @ channel) ** 2
    best_beam = np.argmax(gains)
    best_gain = gains[best_beam]
    return best_beam, best_gain, gains


def exhaustive_search_batch(channels, codebook=None):
    if codebook is None:
        codebook = get_narrow_codebook()
    gains = np.abs(channels @ codebook.conj().T) ** 2
    best_beams = np.argmax(gains, axis=1)
    best_gains = gains[np.arange(len(channels)), best_beams]
    return best_beams, best_gains, gains
