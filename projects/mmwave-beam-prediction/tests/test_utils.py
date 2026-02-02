import numpy as np
import pytest
from beampred.utils import array_response_vector, to_db, from_db, path_loss_umi


class TestToDb:
    def test_basic(self):
        assert to_db(1.0) == pytest.approx(0.0)
        assert to_db(10.0) == pytest.approx(10.0)
        assert to_db(100.0) == pytest.approx(20.0)

    def test_floor(self):
        assert to_db(0.0) == -100
        assert to_db(1e-40) == -100

    def test_custom_floor(self):
        assert to_db(0.0, floor=-50) == -50

    def test_array(self):
        x = np.array([1.0, 10.0, 100.0])
        result = to_db(x)
        np.testing.assert_allclose(result, [0, 10, 20])


class TestFromDb:
    def test_basic(self):
        assert from_db(0) == pytest.approx(1.0)
        assert from_db(10) == pytest.approx(10.0)
        assert from_db(20) == pytest.approx(100.0)

    def test_roundtrip(self):
        x = np.array([0.5, 1.0, 5.0, 50.0])
        np.testing.assert_allclose(from_db(to_db(x)), x, rtol=1e-6)


class TestArrayResponseVector:
    def test_unit_norm(self):
        v = array_response_vector(0.0)
        assert np.abs(np.linalg.norm(v) - 1.0) < 1e-10

    def test_shape(self):
        v = array_response_vector(0.3, n_antennas=32)
        assert v.shape == (32,)

    def test_broadside(self):
        v = array_response_vector(0.0, n_antennas=4)
        np.testing.assert_allclose(np.abs(v), 1.0 / 2.0)

    def test_complex(self):
        v = array_response_vector(0.5)
        assert np.iscomplexobj(v)


class TestPathLossUmi:
    def test_los_vs_nlos(self):
        d = 100.0
        pl_los = path_loss_umi(d, los=True)
        pl_nlos = path_loss_umi(d, los=False)
        assert pl_nlos > pl_los

    def test_increases_with_distance(self):
        pl_near = path_loss_umi(10.0)
        pl_far = path_loss_umi(200.0)
        assert pl_far > pl_near

    def test_min_distance_clamp(self):
        pl = path_loss_umi(0.1)
        pl_one = path_loss_umi(1.0)
        assert pl == pl_one
