"""Check validation selection, tie-breaking and non-finite rejection."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from proposal_completion import choose_a


def row(a, recall):
    return dict(a=a, metrics=dict(recall_at_20=recall))


assert choose_a([row(0, .1), row(.25, .3), row(.5, .2)])["a"] == .25
assert choose_a([row(0, .3), row(.25, .3), row(.5, .3)])["a"] == .5
assert choose_a([row(0, .3), row(.25, .3)])["a"] == .25
for candidates in [[], [row(.5, math.nan)], [row(.25, math.inf)]]:
    try:
        choose_a(candidates)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid selection input was accepted")
print("PASS: validation-only selection, exact tie rule, non-finite rejection")
