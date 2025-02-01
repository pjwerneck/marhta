from typing import List
from typing import Optional
from typing import Tuple

def levenshtein_distance(s1: str, s2: str, cutoff: Optional[int] = None) -> int:
    """Calculate the Levenshtein edit distance between two strings

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string into another.

    Args:
        s1: First string to compare
        s2: Second string to compare
        cutoff: Maximum distance to calculate, returns cutoff + 1 if exceeded

    Returns:
        The edit distance between the strings, or cutoff + 1 if specified and exceeded
    """
    ...

def levenshtein_similarity(s1: str, s2: str, cutoff: Optional[float] = None) -> float:
    """Calculate the Levenshtein similarity between two strings

    The Levenshtein similarity is the inverse of the Levenshtein distance,
    normalized to a value between 0.0 (completely different) and 1.0 (identical).

    Args:
        s1: First string to compare
        s2: Second string to compare
        cutoff: Minimum similarity required, stops early if impossible to reach

    Returns:
        The similarity score between the strings (0.0 to 1.0)
    """
    ...

def levenshtein_match(
    pattern: str, strings: List[str], min: float = 0.0, max: float = 1.0, limit: int = 5
) -> List[Tuple[str, float]]:
    """Find the best Levenshtein matches for a pattern in a list of strings

    Args:
        pattern: The string pattern to match against
        strings: List of strings to search through
        min: Minimum similarity score (0.0 to 1.0)
        max: Maximum similarity score (0.0 to 1.0)
        limit: Maximum number of results to return

    Returns:
        List of tuples containing (matched_string, similarity_score), sorted by score descending
    """
    ...

def jaro_winkler_similarity(
    s1: str, s2: str, prefix_weight: float = 0.1, max_prefix: int = 4
) -> float:
    """Calculate the Jaro-Winkler similarity between two strings

    The Jaro-Winkler similarity is a measure of similarity between two strings.
    The higher the Jaro-Winkler similarity for two strings is, the more similar
    the strings are.

    Args:
        s1: First string to compare
        s2: Second string to compare
        prefix_weight: Weight for the common prefix (0.0 to 0.25)
        max_prefix: Maximum prefix length to consider

    Returns:
        The Jaro-Winkler similarity between the strings
    """
    ...

def jaro_winkler_distance(
    s1: str, s2: str, prefix_weight: float = 0.1, max_prefix: int = 4
) -> float:
    """Calculate the Jaro-Winkler edit distance between two strings

    The Jaro-Winkler distance is a measure of similarity between two strings.
    The lower the Jaro-Winkler distance for two strings is, the more similar the
    strings are.

    Args:
        s1: First string to compare
        s2: Second string to compare
        prefix_weight: Weight for the common prefix (0.0 to 0.25)
        max_prefix: Maximum prefix length to consider

    Returns:
        The Jaro-Winkler distance between the strings
    """
    ...

def jaro_winkler_match(
    pattern: str,
    strings: List[str],
    min: float = 0.0,
    max: float = 1.0,
    limit: int = 5,
    prefix_weight: float = 0.1,
    max_prefix: int = 4,
) -> List[Tuple[str, float]]:
    """Find the best Jaro-Winkler matches for a pattern in a list of strings

    Args:
        pattern: The string pattern to match against
        strings: List of strings to search through
        min: Minimum similarity score (0.0 to 1.0)
        max: Maximum similarity score (0.0 to 1.0)
        limit: Maximum number of results to return
        prefix_weight: Weight for the common prefix (0.0 to 0.25)
        max_prefix: Maximum prefix length to consider

    Returns:
        List of tuples containing (matched_string, similarity_score), sorted by score descending
    """
    ...
