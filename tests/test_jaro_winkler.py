import marhta
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from .fixtures import WORDS
from .fixtures import jaro_winkler_distance
from .fixtures import jaro_winkler_match
from .fixtures import jaro_winkler_similarity


@given(
    st.sampled_from(WORDS),
    st.sampled_from(WORDS),
    st.sampled_from([0.0, 0.1, 0.2, 0.25]),
    st.sampled_from([0, 1, 2, 3, 4]),
)
@settings(max_examples=1000)
def test_jaro_winkler_similarity_with_sane_parameters(s1, s2, prefix_weight, max_prefix):
    expected = jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)
    result = marhta.jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)
    assert 0 <= result <= 1
    assert result == expected


@given(
    st.text(min_size=0, max_size=4),
    st.text(),
    st.text(),
    st.floats(min_value=0, max_value=0.25),
    st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=1000)
def test_jaro_winkler_similarity_sanity(prefix, s1, s2, prefix_weight, max_prefix):
    s1 = prefix + s1
    s2 = prefix + s2
    expected = jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)
    result = marhta.jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)
    assert 0 <= result <= 1
    assert result == expected


@given(
    st.text(min_size=0, max_size=4),
    st.text(),
    st.text(),
    st.floats(min_value=0, max_value=0.25),
    st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=1000)
def test_jaro_winkler_distance_sanity(prefix, s1, s2, prefix_weight, max_prefix):
    s1 = prefix + s1
    s2 = prefix + s2
    expected = jaro_winkler_distance(s1, s2, prefix_weight, max_prefix)
    result = marhta.jaro_winkler_distance(s1, s2, prefix_weight, max_prefix)
    assert 0 <= result <= 1
    assert result == expected


@given(
    st.text(),
    st.lists(st.text()),
    st.floats(min_value=0, max_value=1).flatmap(
        lambda x: st.tuples(st.just(x), st.floats(min_value=x, max_value=1))
    ),
    st.integers(min_value=0, max_value=2**32 - 1),
    st.floats(min_value=0, max_value=0.25),
    st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=1000)
def test_jaro_winkler_match_sanity(pattern, strings, min_max, limit, prefix_weight, max_prefix):
    min, max = min_max
    expected = jaro_winkler_match(pattern, strings, min, max, limit, prefix_weight, max_prefix)
    result = marhta.jaro_winkler_match(
        pattern, strings, min, max, limit, prefix_weight, max_prefix
    )

    assert result == expected
