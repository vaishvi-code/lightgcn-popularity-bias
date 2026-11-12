"""Check tied ranks, constant inputs, input validation, and the diagnostic mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from embedding_metrics import correlation, embedding_diagnostics

np.testing.assert_allclose(correlation([1, 1, 2, 3], [10, 10, 20, 30]), 1)
np.testing.assert_allclose(correlation([1, 2, 3], [3, 2, 1]), -1)
assert correlation([1, 1, 1], [1, 2, 3]) is None
for left, right in [([1], [2]), ([1, 2], [3]), ([1, float("nan")], [1, 2])]:
    try:
        correlation(left, right)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid diagnostic input accepted")
degree = np.array([0, 1, 4, 9])
norm = np.sqrt(degree)
observed = embedding_diagnostics(degree, norm, norm, norm, degree)
for value in observed.values():
    np.testing.assert_allclose(value, 1)
print("PASS: average tied ranks, reverse ranks, undefined constant correlation, invalid inputs, and degree/norm/exposure diagnostics.")
