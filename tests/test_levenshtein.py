import json

import pytest
from marhta import levenshtein_distance


@pytest.mark.parametrize(
    "s1, s2, expected",
    json.load(open("tests/fixtures/levenshtein_distance.json")),
)
def test_levenshtein_distance(s1, s2, expected):
    assert levenshtein_distance(s1, s2) == expected
