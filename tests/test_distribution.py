import pytest

from bocd.distribution import Distribution


class TestDistribution:
    def test_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            Distribution().reset_params()

        with pytest.raises(NotImplementedError):
            Distribution().pdf(100)
        with pytest.raises(NotImplementedError):
            Distribution().update_params(100)
