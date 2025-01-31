import json

import pytest
from marhta import jaro_winkler_similarity


@pytest.mark.parametrize(
    "s1, s2, expected",
    json.load(open("tests/fixtures/jaro_winkler_similarity.json")),
)
def test_jaro_winkler_similarity(s1, s2, expected):
    assert jaro_winkler_similarity(s1, s2, 0.1, 4) == expected
