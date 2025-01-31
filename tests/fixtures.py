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


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    result = previous_row[-1]
    assert 0 <= result
    return result


def sanity_check():
    assert levenshtein_distance("a", "a") == 0
    assert levenshtein_distance("a", "b") == 1
    assert levenshtein_distance("MARTHA", "MARTHA") == 0
    assert levenshtein_distance("MARTHA", "MARHTA") == 2

    assert jaro("a", "a") == 1.0
    assert jaro("a", "b") == 0.0
    assert jaro_winkler_similarity("MARTHA", "MARTHA") == 1.0
    assert jaro_winkler_similarity("MARTHA", "MARHTA") == 0.9611111111111111
    assert jaro_winkler_similarity("gloater", "biometrical") == 0.5616883116883117, (
        jaro_winkler_similarity("gloater", "biometrical")
    )


sanity_check()
