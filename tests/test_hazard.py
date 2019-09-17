import pytest

from bocd.hazard import ConstantHazard, Hazard


class TestHazard:
    def test_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            x = Hazard()
            x()


class TestConstant:
    def test_lambda(self):
        ch = ConstantHazard(10)
        assert ch(123) == 1 / 10
