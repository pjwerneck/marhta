use pyo3::prelude::*;
use std::cmp::min;

// GIL release threshold in characters - Levenshtein is O(m*n)
const LEVENSHTEIN_GIL_RELEASE_THRESHOLD: usize = 64;

/// Calculate the actual distance, with optional early stopping
fn _levenshtein_distance(s1: &str, s2: &str, cutoff: Option<usize>) -> usize {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    // Early returns for empty strings - but respect cutoff!
    if s1_len == 0 {
        return if let Some(max_dist) = cutoff {
            min(s2_len, max_dist + 1)
        } else {
            s2_len
        };
    }
    if s2_len == 0 {
        return if let Some(max_dist) = cutoff {
            min(s1_len, max_dist + 1)
        } else {
            s1_len
        };
    }

    // Quick check if absolute length difference exceeds cutoff
    if let Some(max_dist) = cutoff {
        if s1_len.abs_diff(s2_len) > max_dist {
            return max_dist + 1; // Return value larger than cutoff
        }
    }

    let mut prev_row: Vec<usize> = (0..=s2_len).collect();
    let mut current_row = vec![0; s2_len + 1];

    for (i, c1) in s1.chars().enumerate() {
        current_row[0] = i + 1;
        let mut min_dist = current_row[0];

        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            current_row[j + 1] = min(
                min(current_row[j] + 1, prev_row[j + 1] + 1),
                prev_row[j] + cost,
            );
            min_dist = min(min_dist, current_row[j + 1]);
        }

        // Early stopping check - if entire row exceeds cutoff
        if let Some(max_dist) = cutoff {
            if min_dist > max_dist {
                return max_dist + 1; // Return value larger than cutoff
            }
        }

        std::mem::swap(&mut prev_row, &mut current_row);
    }

    // Return minimum of final distance and cutoff + 1 if cutoff exists
    if let Some(max_dist) = cutoff {
        min(prev_row[s2_len], max_dist + 1)
    } else {
        prev_row[s2_len]
    }
}

/// Calculate similarity with optional early stopping
fn _levenshtein_similarity(s1: &str, s2: &str, cutoff: Option<f64>) -> f64 {
    let max_len = s1.chars().count().max(s2.chars().count());
    if max_len == 0 {
        return 1.0;
    }

    // Convert similarity cutoff to distance cutoff
    let distance_cutoff = if let Some(min_similarity) = cutoff {
        Some((1.0 - min_similarity) * max_len as f64).map(|x| x.ceil() as usize)
    } else {
        None
    };

    let distance = _levenshtein_distance(s1, s2, distance_cutoff);
    1.0 - (distance as f64 / max_len as f64)
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
        // Use min as cutoff - no need to calculate exact distance if we know
        // it won't meet the minimum similarity requirement
        let score = _levenshtein_similarity(pattern, &s, Some(actual_min));
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
#[pyo3(signature = (s1, s2, cutoff = None))]
/// Calculate the Levenshtein edit distance between two strings
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string into another.
///
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
///     cutoff (Optional[int]): Maximum distance to calculate, returns cutoff + 1 if exceeded
///
/// Returns:
///     int: The edit distance between the strings, or cutoff + 1 if specified and exceeded
pub fn levenshtein_distance(s1: &str, s2: &str, cutoff: Option<usize>) -> PyResult<usize> {
    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD || s2_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD {
        Python::with_gil(|py| py.allow_threads(|| Ok(_levenshtein_distance(s1, s2, cutoff))))
    } else {
        Ok(_levenshtein_distance(s1, s2, cutoff))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, cutoff = None))]
/// Calculate the Levenshtein similarity between two strings
///
/// The Levenshtein similarity is the inverse of the Levenshtein distance,
/// normalized to a value between 0.0 (completely different) and 1.0 (identical).
///
/// Args:
///     s1 (str): First string to compare    
///     s2 (str): Second string to compare
///     cutoff (Optional[float]): Minimum similarity required, stops early if impossible to reach
///
/// Returns:
///     float: The similarity score between the strings (0.0 to 1.0)
pub fn levenshtein_similarity(s1: &str, s2: &str, cutoff: Option<f64>) -> PyResult<f64> {
    if let Some(c) = cutoff {
        if !(0.0..=1.0).contains(&c) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cutoff must be between 0.0 and 1.0",
            ));
        }
    }

    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD || s2_len > LEVENSHTEIN_GIL_RELEASE_THRESHOLD {
        Python::with_gil(|py| py.allow_threads(|| Ok(_levenshtein_similarity(s1, s2, cutoff))))
    } else {
        Ok(_levenshtein_similarity(s1, s2, cutoff))
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
        assert_eq!(_levenshtein_distance("martha", "marhta", None), 2);
        assert_eq!(_levenshtein_distance("kitten", "sitting", None), 3);
        assert_eq!(_levenshtein_distance("saturday", "sunday", None), 3);
        assert_eq!(_levenshtein_distance("", "", None), 0);
        assert_eq!(_levenshtein_distance("abc", "", None), 3);
        assert_eq!(_levenshtein_distance("", "xyz", None), 3);
        assert_eq!(_levenshtein_distance("abc", "abc", None), 0);

        // Edge cases
        assert_eq!(_levenshtein_distance("a", "", None), 1);
        assert_eq!(_levenshtein_distance("", "a", None), 1);
        assert_eq!(_levenshtein_distance("abc", "acb", None), 2);
        assert_eq!(_levenshtein_distance("abc", "bca", None), 2);
        assert_eq!(
            _levenshtein_distance(&"a".repeat(1000), &"b".repeat(1000), None),
            1000
        );
        // TODO: test with larger strings, 1MB or more

        // Unicode handling
        assert_eq!(_levenshtein_distance("café", "cafe", None), 1);
        assert_eq!(_levenshtein_distance("こんにちは", "konnichiwa", None), 10);
    }

    #[test]
    fn test_distance_with_cutoff() {
        assert_eq!(_levenshtein_distance("kitten", "sitting", Some(2)), 3);
        assert_eq!(_levenshtein_distance("kitten", "sitting", Some(1)), 2);
        assert_eq!(_levenshtein_distance("abc", "def", Some(2)), 3);
        assert_eq!(_levenshtein_distance("short", "verylongstring", Some(3)), 4);
        assert_eq!(_levenshtein_distance("prefixabc", "prefixdef", Some(2)), 3);
        // these cases were from failing hypothesis tests
        assert_eq!(_levenshtein_distance("1080", "10-point", Some(4)), 5);
        assert_eq!(_levenshtein_distance("1080", "10-point", Some(3)), 4);
        assert_eq!(_levenshtein_distance("short", "verylongstring", Some(4)), 5);
        assert_eq!(_levenshtein_distance("aasvogel", "selsyn", Some(5)), 6);
    }

    #[test]
    fn test_similarity() {
        assert_eq!(_levenshtein_similarity("", "", None), 1.0);
        assert_eq!(
            _levenshtein_similarity("kitten", "sitting", None),
            1.0 - 3.0 / 7.0
        );
        assert_eq!(_levenshtein_similarity("abc", "xyz", None), 0.0);
    }

    #[test]
    fn test_similarity_with_cutoff() {
        // Should calculate exact similarity
        assert_eq!(
            _levenshtein_similarity("kitten", "sitting", Some(0.5)),
            1.0 - 3.0 / 7.0
        );

        // Should stop early - similarity can't reach cutoff
        assert!(_levenshtein_similarity("abc", "xyz", Some(0.9)) < 0.9);

        // Should get exact match for high similarity strings
        assert_eq!(_levenshtein_similarity("abcdef", "abcdef", Some(0.9)), 1.0);
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

    #[test]
    fn test_match_with_cutoff() {
        let strings = vec![
            "kitten".to_string(),
            "sitting".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        // Should only include matches above 0.8 similarity
        let matches = _levenshtein_match("kitten", strings, 0.8, 1.0, 10);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, "kitten");
    }
}
