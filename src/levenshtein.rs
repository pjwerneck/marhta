use pyo3::prelude::*;
use std::cmp::min;

// GIL release threshold in characters - Levenshtein is O(m*n)
const LEVENSHTEIN_GIL_RELEASE_THRESHOLD: usize = 64;

/// Calculate the actual distance
fn _levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len == 0 {
        return s2_len;
    }
    if s2_len == 0 {
        return s1_len;
    }

    // Use character counts instead of byte lengths
    let mut prev_row: Vec<usize> = (0..=s2_len).collect();
    let mut current_row = vec![0; s2_len + 1];

    for (i, c1) in s1.chars().enumerate() {
        current_row[0] = i + 1;
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            current_row[j + 1] = min(
                min(current_row[j] + 1, prev_row[j + 1] + 1),
                prev_row[j] + cost,
            );
        }
        std::mem::swap(&mut prev_row, &mut current_row);
    }

    prev_row[s2_len]
}

/// Calculate similarity
fn _levenshtein_similarity(s1: &str, s2: &str) -> f64 {
    let distance = _levenshtein_distance(s1, s2);
    let max_len = s1.chars().count().max(s2.chars().count());

    if max_len == 0 {
        1.0
    } else {
        1.0 - (distance as f64 / max_len as f64)
    }
}

// Calculate the best matches
fn _levenshtein_match(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
) -> Vec<(String, f64)> {
    let (actual_min, actual_max) = if min <= max { (min, max) } else { (max, min) };
    let mut matches = Vec::with_capacity(strings.len());

    for s in strings {
        let score = _levenshtein_similarity(pattern, &s);
        if score >= actual_min && score <= actual_max {
            matches.push((s, score));
        }
    }

    matches.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    matches.into_iter().take(limit).collect()
}

#[pyfunction]
/// Calculate the Levenshtein edit distance between two strings
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string into another.
///
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
///
/// Returns:
///     int: The edit distance between the strings
pub fn levenshtein_distance(s1: &str, s2: &str) -> PyResult<usize> {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD || s2_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD {
        Python::with_gil(|py| py.allow_threads(|| Ok(_levenshtein_distance(s1, s2))))
    } else {
        Ok(_levenshtein_distance(s1, s2))
    }
}

#[pyfunction]
/// Calculate the Levenshtein similarity between two strings
///
/// The Levenshtein similarity is the inverse of the Levenshtein distance,
/// normalized to a value between 0.0 (completely different) and 1.0
/// (identical).
///
/// Args:
///     s1 (str): First string to compare    
///     s2 (str): Second string to compare
///
/// Returns:
///     float: The similarity score between the strings (0.0 to 1.0)
pub fn levenshtein_similarity(s1: &str, s2: &str) -> PyResult<f64> {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD || s2_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD {
        Python::with_gil(|py| py.allow_threads(|| Ok(_levenshtein_similarity(s1, s2))))
    } else {
        Ok(_levenshtein_similarity(s1, s2))
    }
}

/// Find the best Levenshtein matches for a pattern in a list of strings
///
/// Args:
///     pattern (str): The string pattern to match against
///     strings (List[str]): List of strings to search through
///     min (float, optional): Minimum similarity score (0.0 to 1.0). Defaults to 0.0
///     max (float, optional): Maximum similarity score (0.0 to 1.0). Defaults to 1.0
///     limit (int, optional): Maximum number of results to return. Defaults to 5
///
/// Returns:
///     List[Tuple[str, float]]: List of tuples containing (matched_string, similarity_score),
///     sorted by score descending
#[pyfunction]
#[pyo3(signature = (pattern, strings, min = 0.0, max = 1.0, limit = 5))]
pub fn levenshtein_match(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
) -> PyResult<Vec<(String, f64)>> {
    Ok(_levenshtein_match(pattern, strings, min, max, limit))
}

// Basic tests to ensure the functions work as expected. Extensive tests are in
// the Python test suite.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        // Base cases
        assert_eq!(_levenshtein_distance("martha", "marhta"), 2);
        assert_eq!(_levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(_levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(_levenshtein_distance("", ""), 0);
        assert_eq!(_levenshtein_distance("abc", ""), 3);
        assert_eq!(_levenshtein_distance("", "xyz"), 3);
        assert_eq!(_levenshtein_distance("abc", "abc"), 0);

        // Edge cases
        assert_eq!(_levenshtein_distance("a", ""), 1);
        assert_eq!(_levenshtein_distance("", "a"), 1);
        assert_eq!(_levenshtein_distance("abc", "acb"), 2);
        assert_eq!(_levenshtein_distance("abc", "bca"), 2);
        assert_eq!(
            _levenshtein_distance(&"a".repeat(1000), &"b".repeat(1000)),
            1000
        );
        // TODO: test with larger strings, 1MB or more

        // Unicode handling
        assert_eq!(_levenshtein_distance("café", "cafe",), 1);
        assert_eq!(_levenshtein_distance("こんにちは", "konnichiwa",), 10);
    }

    #[test]
    fn test_similarity() {
        assert_eq!(_levenshtein_similarity("", ""), 1.0);
        assert_eq!(
            _levenshtein_similarity("kitten", "sitting"),
            1.0 - 3.0 / 7.0
        );
        assert_eq!(_levenshtein_similarity("abc", "xyz"), 0.0);
    }

    #[test]
    fn test_match() {
        let strings = vec![
            "kitten".to_string(),
            "sitting".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        let matches = _levenshtein_match("kitten", strings, 0.0, 1.0, 2);

        assert!((matches[0].1 - 1.0).abs() < f64::EPSILON);
        assert_eq!(matches[0].0, "kitten");

        assert!((matches[1].1 - (1.0 - 3.0 / 7.0)).abs() < f64::EPSILON);
        assert_eq!(matches[1].0, "sitting");
    }
}
