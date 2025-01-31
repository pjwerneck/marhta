from pathlib import Path

HERE = Path(__file__).parent

with open(HERE / "words.txt") as f:
    WORDS = f.read().splitlines()

# These are reference implementations of the algorithms, verified against a
# third-party implementation. They are used to generate test cases for the
# actual rust implementation.


def jaro(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Match distance is floor(max(|A|,|B|)/2) - 1
    match_distance = max(len(s1), len(s2)) // 2 - 1

    # Ensure match_distance is at least 0
    match_distance = max(0, match_distance)

    s1_matches = []  # Indexes of matched characters in s1
    s2_matches = []  # Indexes of matched characters in s2

    # Find matches within match distance
    for i, ch1 in enumerate(s1):
        # Calculate matching window
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))

        for j in range(start, end):
            if j not in s2_matches and ch1 == s2[j]:
                s1_matches.append(i)
                s2_matches.append(j)
                break

    if not s1_matches:
        return 0.0

    # Count transpositions
    transpositions = 0
    for s1_pos, s2_pos in zip(sorted(s1_matches), sorted(s2_matches)):
        if s1[s1_pos] != s2[s2_pos]:
            transpositions += 1

    m = len(s1_matches)
    return ((m / len(s1)) + (m / len(s2)) + ((m - transpositions / 2) / m)) / 3.0


def jaro_winkler_similarity(s1, s2, p=0.1, max_prefix=4):
    # Get basic Jaro distance
    jaro_dist = jaro(s1, s2)

    # Get length of common prefix up to max_prefix chars
    prefix_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2 or prefix_len >= max_prefix:
            break
        prefix_len += 1

    # Apply Winkler modification
    result = jaro_dist + (prefix_len * p * (1.0 - jaro_dist))
    return min(1.0, result)


def levenshtein_distance(s1, s2, cutoff=None):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1, cutoff)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1) if cutoff is None else min(len(s1), cutoff + 1)

    # Quick check if absolute length difference exceeds cutoff
    if cutoff is not None and abs(len(s1) - len(s2)) > cutoff:
        return cutoff + 1

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        min_dist = current_row[0]

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current = min(insertions, deletions, substitutions)
            current_row.append(current)
            min_dist = min(min_dist, current)

        # Early stopping check
        if cutoff is not None and min_dist > cutoff:
            return cutoff + 1

        previous_row = current_row

    # Return minimum of final distance and cutoff + 1 if cutoff exists
    return previous_row[-1] if cutoff is None else min(previous_row[-1], cutoff + 1)


def levenshtein_similarity(s1, s2, cutoff=None):
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    # Convert similarity cutoff to distance cutoff
    if cutoff is not None:
        distance_cutoff = int((1.0 - cutoff) * max_len + 0.999999)
    else:
        distance_cutoff = None

    distance = levenshtein_distance(s1, s2, distance_cutoff)
    return 1.0 - (distance / max_len)


def levenshtein_match(pattern, strings, min=0.0, max=1.0, limit=5):
    matches = []
    for s in strings:
        # Use min as cutoff for early stopping
        score = levenshtein_similarity(pattern, s, min)
        if min <= score <= max:
            matches.append((s, score))

    # Sort by score descending, then by string lexicographically
    matches.sort(key=lambda x: (-x[1], x[0]))
    return matches[:limit]


def jaro_winkler_distance(s1, s2, prefix_weight=0.1, max_prefix=4):
    return 1.0 - jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)


def jaro_winkler_match(
    pattern, strings, min=0.0, max=1.0, limit=5, prefix_weight=0.1, max_prefix=4
):
    matches = []
    for s in strings:
        score = jaro_winkler_similarity(pattern, s, prefix_weight, max_prefix)
        if min <= score <= max:
            matches.append((s, score))

    # Sort by score descending, then by string lexicographically
    matches.sort(key=lambda x: (-x[1], x[0]))
    return matches[:limit]


def sanity_check():
    # Levenshtein tests
    assert levenshtein_distance("a", "a") == 0
    assert levenshtein_distance("a", "b") == 1
    assert levenshtein_distance("MARTHA", "MARTHA") == 0
    assert levenshtein_distance("MARTHA", "MARHTA") == 2
    assert levenshtein_similarity("kitten", "sitting") == 1.0 - 3 / 7
    assert levenshtein_match("kitten", ["kitten", "sitting"], min=0.5)[0][0] == "kitten"

    # Early stopping tests
    assert levenshtein_distance("abcd", "defg", 2) == 3
    assert levenshtein_similarity("abcd", "defg", 0.9) < 0.9

    # Jaro-Winkler tests
    assert jaro("a", "a") == 1.0
    assert jaro("a", "b") == 0.0
    assert jaro_winkler_similarity("MARTHA", "MARTHA") == 1.0
    assert jaro_winkler_similarity("MARTHA", "MARHTA") == 0.9611111111111111
    assert jaro_winkler_similarity("gloater", "biometrical") == 0.5616883116883117

    # Test clamping to 1.0
    assert jaro_winkler_similarity("0000", "00000", 0.25, 5) == 1.0


sanity_check()
