import numpy as np
from beampred.codebook import generate_dft_codebook, get_narrow_codebook, get_wide_codebook, beam_angles


class TestGenerateDftCodebook:
    def test_shape(self):
        cb = generate_dft_codebook(64, 16)
        assert cb.shape == (16, 64)

    def test_unit_norm_rows(self):
        cb = generate_dft_codebook(32, 8)
        norms = np.linalg.norm(cb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_orthogonal_when_square(self):
        cb = generate_dft_codebook(8, 8)
        inner = cb @ cb.conj().T
        np.testing.assert_allclose(np.abs(inner), np.eye(8), atol=1e-10)


class TestGetCodebooks:
    def test_narrow_shape(self):
        cb = get_narrow_codebook()
        assert cb.shape == (64, 64)

    def test_wide_shape(self):
        cb = get_wide_codebook()
        assert cb.shape == (16, 64)


class TestBeamAngles:
    def test_count(self):
        angles = beam_angles(16)
        assert len(angles) == 16

    def test_symmetric(self):
        angles = beam_angles(16)
        np.testing.assert_allclose(angles + angles[::-1], 0.0, atol=1e-10)

    def test_range(self):
        angles = beam_angles(64)
        assert angles.min() > -np.pi / 2
        assert angles.max() < np.pi / 2
