import marhta
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from .fixtures import WORDS
from .fixtures import levenshtein_distance
from .fixtures import levenshtein_match
from .fixtures import levenshtein_similarity


@given(
    st.sampled_from(WORDS), st.sampled_from(WORDS), st.integers(min_value=0, max_value=2**32 - 1)
)
@settings(max_examples=1000)
def test_levenshtein_distance_with_sane_parameters(s1, s2, cutoff):
    expected = levenshtein_distance(s1, s2, cutoff=cutoff)
    result = marhta.levenshtein_distance(s1, s2, cutoff=cutoff)
    assert 0 <= result
    assert result == expected


@given(st.text(), st.text())
@settings(max_examples=1000)
def test_levenshtein_distance_sanity(s1, s2):
    expected = levenshtein_distance(s1, s2)
    result = marhta.levenshtein_distance(s1, s2)
    assert 0 <= result
    assert result == expected


@given(st.text(), st.text())
@settings(max_examples=1000)
def test_levenshtein_similarity_sanity(s1, s2):
    expected = levenshtein_similarity(s1, s2)
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
    expected = levenshtein_match(pattern, strings, min, max, limit)
    result = marhta.levenshtein_match(pattern, strings, min, max, limit)

    assert result == expected
