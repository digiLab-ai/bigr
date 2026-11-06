"""Import-smoke coverage for the public BIGR surface."""

import bigr
from bigr import BIGRFilter, InferenceConfig, StochasticFTheta


def test_public_imports_smoke() -> None:
    assert BIGRFilter.__name__ == "BIGRFilter"
    assert InferenceConfig().particles == 16
    assert StochasticFTheta.__name__ == "StochasticFTheta"
    assert "BIGRFilter" in bigr.__all__
