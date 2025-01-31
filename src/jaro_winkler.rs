use pyo3::prelude::*;
use std::cmp::{max, min};

// GIL release threshold in characters - Jaro-Winkler is O(m)
const JARO_WINKLER_GIL_RELEASE_THRESHOLD: usize = 128;

fn _matching_characters(a: &str, b: &str, max_distance: usize) -> (usize, usize) {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut matches = 0;
    let mut transpositions = 0;
    let mut b_matches = vec![false; b_chars.len()];
    let mut match_indexes = Vec::new();

    // Find matches
    for (i, &a_char) in a_chars.iter().enumerate() {
        let start = if i > max_distance {
            i - max_distance
        } else {
            0
        };
        let end = min(i + max_distance + 1, b_chars.len());

        for j in start..end {
            if !b_matches[j] && a_char == b_chars[j] {
                b_matches[j] = true;
                matches += 1;
                match_indexes.push((i, j));
                break;
            }
        }
    }

    // Count transpositions (only counting half as they're counted twice)
    for i in 0..match_indexes.len() {
        for j in i + 1..match_indexes.len() {
            if match_indexes[i].1 > match_indexes[j].1 {
                transpositions += 1;
            }
        }
    }

    (matches, transpositions) // No need to double transpositions anymore
}

fn _jaro_winkler_similarity(s1: &str, s2: &str, prefix_weight: f64, max_prefix: usize) -> f64 {
    if prefix_weight < 0.0 || prefix_weight > 0.25 {
        panic!("prefix_weight must be between 0.0 and 0.25");
    }

    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let max_distance = (max(s1.chars().count(), s2.chars().count()) / 2).saturating_sub(1);
    let (matches, transpositions) = _matching_characters(s1, s2, max_distance);

    if matches == 0 {
        return 0.0;
    }

    let m = matches as f64;
    let t = transpositions as f64; // Already in correct form from _matching_characters
    let s1_len = s1.chars().count() as f64;
    let s2_len = s2.chars().count() as f64;

    // Calculate basic Jaro similarity
    let jaro = (m / s1_len + m / s2_len + (m - t) / m) / 3.0;

    // Calculate common prefix length with configurable limit
    let l = s1
        .chars()
        .zip(s2.chars())
        .take(max_prefix)
        .take_while(|(a, b)| a == b)
        .count() as f64;

    // Apply Winkler modification
    jaro + (l * prefix_weight * (1.0 - jaro))
}

fn _jaro_winkler_distance(s1: &str, s2: &str, prefix_weight: f64, max_prefix: usize) -> f64 {
    1.0 - _jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)
}

fn _jaro_winkler_match(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
    prefix_weight: f64,
    max_prefix: usize,
) -> Vec<(String, f64)> {
    let (actual_min, actual_max) = if min <= max { (min, max) } else { (max, min) };
    let mut matches = Vec::with_capacity(strings.len());

    for s in strings {
        let score = _jaro_winkler_similarity(pattern, &s, prefix_weight, max_prefix);
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
#[pyo3(signature = (s1, s2, prefix_weight = 0.1, max_prefix = 4))]
/// Calculate the Jaro-Winkler similarity between two strings
///
/// The Jaro-Winkler similarity is a measure of similarity between two strings.
/// The higher the Jaro-Winkler similarity for two strings is, the more similar
/// the strings are.
///
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
///     prefix_weight (float, optional): Weight for the common prefix (0.0 to 0.25). Defaults to 0.1
///     max_prefix (int, optional): Maximum prefix length to consider. Defaults to 4
///
/// Returns:
///     float: The Jaro-Winkler similarity between the strings
pub fn jaro_winkler_similarity(
    s1: &str,
    s2: &str,
    prefix_weight: f64,
    max_prefix: usize,
) -> PyResult<f64> {
    if !(0.0..=0.25).contains(&prefix_weight) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prefix_weight must be between 0.0 and 0.25",
        ));
    }

    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len > JARO_WINKLER_GIL_RELEASE_THRESHOLD || s2_len > JARO_WINKLER_GIL_RELEASE_THRESHOLD {
        Python::with_gil(|py| {
            py.allow_threads(|| Ok(_jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix)))
        })
    } else {
        Ok(_jaro_winkler_similarity(s1, s2, prefix_weight, max_prefix))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, prefix_weight = 0.1, max_prefix = 4))]
/// Calculate the Jaro-Winkler edit distance between two strings
///
/// The Jaro-Winkler distance is a measure of similarity between two strings.
/// The lower the Jaro-Winkler distance for two strings is, the more similar the
/// strings are.
///
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
///     prefix_weight (float, optional): Weight for the common prefix (0.0 to 0.25). Defaults to 0.1
///     max_prefix (int, optional): Maximum prefix length to consider. Defaults to 4
///
/// Returns:
///     float: The Jaro-Winkler distance between the strings
pub fn jaro_winkler_distance(
    s1: &str,
    s2: &str,
    prefix_weight: f64,
    max_prefix: usize,
) -> PyResult<f64> {
    if !(0.0..=0.25).contains(&prefix_weight) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prefix_weight must be between 0.0 and 0.25",
        ));
    }

    let s1_len = s1.chars().count();
    let s2_len = s2.chars().count();

    if s1_len > JARO_WINKLER_GIL_RELEASE_THRESHOLD || s2_len > JARO_WINKLER_GIL_RELEASE_THRESHOLD {
        Python::with_gil(|py| {
            py.allow_threads(|| Ok(_jaro_winkler_distance(s1, s2, prefix_weight, max_prefix)))
        })
    } else {
        Ok(_jaro_winkler_distance(s1, s2, prefix_weight, max_prefix))
    }
}

#[pyfunction]
#[pyo3(signature = (pattern, strings, min = 0.0, max = 1.0, limit = 5, prefix_weight = 0.1, max_prefix = 4))]
/// Find the best Jaro-Winkler matches for a pattern in a list of strings
///
/// Args:
///     pattern (str): The string pattern to match against
///     strings (List[str]): List of strings to search through
///     min (float, optional): Minimum similarity score (0.0 to 1.0). Defaults to 0.0
///     max (float, optional): Maximum similarity score (0.0 to 1.0). Defaults to 1.0
///     limit (int, optional): Maximum number of results to return. Defaults to 5
///     prefix_weight (float, optional): Weight for the common prefix (0.0 to 0.25). Defaults to 0.1
///     max_prefix (int, optional): Maximum prefix length to consider. Defaults to 4
///
/// Returns:
///     List[Tuple[str, float]]: List of tuples containing (matched_string, similarity_score),
///     sorted by score descending
pub fn jaro_winkler_match(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
    prefix_weight: f64,
    max_prefix: usize,
) -> PyResult<Vec<(String, f64)>> {
    if !(0.0..=0.25).contains(&prefix_weight) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prefix_weight must be between 0.0 and 0.25",
        ));
    }

    Ok(_jaro_winkler_match(
        pattern,
        strings,
        min,
        max,
        limit,
        prefix_weight,
        max_prefix,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_jaro_winkler() {
        // Standard test cases (existing)
        assert_relative_eq!(
            _jaro_winkler_similarity("MARTHA", "MARHTA", 0.1, 4),
            0.961,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("DWAYNE", "DUANE", 0.1, 4),
            0.840,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("ABCD", "EFGH", 0.1, 4),
            0.0,
            epsilon = 0.001
        );

        // Base cases
        assert_relative_eq!(
            _jaro_winkler_similarity("kitten", "sitting", 0.1, 4),
            0.746,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("saturday", "sunday", 0.1, 4),
            0.7175,
            epsilon = 0.001
        );
        assert_eq!(_jaro_winkler_similarity("", "", 0.1, 4), 1.0);
        assert_eq!(_jaro_winkler_similarity("abc", "", 0.1, 4), 0.0);
        assert_eq!(_jaro_winkler_similarity("", "xyz", 0.1, 4), 0.0);
        assert_eq!(_jaro_winkler_similarity("abc", "abc", 0.1, 4), 1.0);

        // Edge cases
        assert_eq!(_jaro_winkler_similarity("test", "", 0.1, 4), 0.0);
        assert_eq!(_jaro_winkler_similarity("", "test", 0.1, 4), 0.0);
        assert_relative_eq!(
            _jaro_winkler_similarity("abc", "acb", 0.1, 4),
            0.5999,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("abc", "bca", 0.1, 4),
            0.0,
            epsilon = 0.001
        );

        // Large string test
        let long_a = "a".repeat(1000);
        let long_b = "b".repeat(1000);
        assert_eq!(_jaro_winkler_similarity(&long_a, &long_b, 0.1, 4), 0.0);

        // Unicode handling
        assert_relative_eq!(
            _jaro_winkler_similarity("café", "cafe", 0.1, 4),
            0.883,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("こんにちは", "konnichiwa", 0.1, 4),
            0.000,
            epsilon = 0.001
        );

        // Test different prefix weights
        assert_relative_eq!(
            _jaro_winkler_similarity("MARTHA", "MARHTA", 0.0, 4),
            0.944,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("MARTHA", "MARHTA", 0.1, 4),
            0.961,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("MARTHA", "MARHTA", 0.2, 4),
            0.977,
            epsilon = 0.001
        );
    }

    #[test]
    fn test_jaro_winkler_match() {
        let strings = vec![
            "apple".to_string(),
            "apples".to_string(),
            "aple".to_string(),
            "appliance".to_string(),
        ];

        let result = _jaro_winkler_match("apple", strings, 0.0, 1.0, 4, 0.1, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].0, "apple");
        assert_relative_eq!(result[0].1, 1.0, epsilon = 0.001);
        assert_eq!(result[1].0, "apples");
        assert_relative_eq!(result[1].1, 0.966, epsilon = 0.001);
    }

    #[test]
    fn test_max_prefix() {
        assert_relative_eq!(
            _jaro_winkler_similarity("prefix", "prefixx", 0.1, 4),
            0.971,
            epsilon = 0.001
        );
        assert_relative_eq!(
            _jaro_winkler_similarity("prefix", "prefixx", 0.1, 6),
            0.980,
            epsilon = 0.001
        );
    }
}
