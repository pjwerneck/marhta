import marhta
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from .fixtures import WORDS
from .fixtures import levenshtein_distance as fx_levenshtein_distance


@given(st.sampled_from(WORDS), st.sampled_from(WORDS))
@settings(max_examples=1000)
def test_levenshtein_distance_with_sane_parameters(s1, s2):
    expected = fx_levenshtein_distance(s1, s2)
    result = marhta.levenshtein_distance(s1, s2)
    assert 0 <= result
    assert result == expected


@given(st.text(), st.text())
@settings(max_examples=1000)
def test_levenshtein_distance_sanity(s1, s2):
    expected = fx_levenshtein_distance(s1, s2)
    result = marhta.levenshtein_distance(s1, s2)
    assert 0 <= result
    assert result == expected


@given(st.text(), st.text())
@settings(max_examples=1000)
def test_levenshtein_similarity_sanity(s1, s2):
    n = max(len(s1), len(s2))
    expected = 1 if not n else 1 - fx_levenshtein_distance(s1, s2) / n
    result = marhta.levenshtein_similarity(s1, s2)
    assert 0 <= result <= 1
    assert result == expected


@given(
    st.text(),
    st.lists(st.text(), min_size=0, max_size=10),
    st.floats(min_value=0, max_value=1).flatmap(
        lambda x: st.tuples(st.just(x), st.floats(min_value=x, max_value=1))
    ),
    st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=1000)
def test_levenshtein_match_sanity(pattern, strings, min_max, limit):
    min, max = min_max
    expected = [(s, marhta.levenshtein_similarity(pattern, s)) for s in strings]
    # Sort by score descending, then by string lexicographically
    expected = sorted(expected, key=lambda x: (-x[1], x[0]))
    expected = [x for x in expected if min <= x[1] <= max]
    expected = expected[:limit]

    result = marhta.levenshtein_match(pattern, strings, min, max, limit)

    assert result == expected
