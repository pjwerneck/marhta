use pyo3::prelude::*;
use std::cmp::min;

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
#[pyfunction]
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
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

/// Calculate the normalized Levenshtein similarity between two strings
/// 
/// This is calculated as 1 - (distance / max_length), resulting in a score
/// between 0.0 (completely different) and 1.0 (exact match).
/// 
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
/// 
/// Returns:
///     float: Similarity score between 0.0 and 1.0
#[pyfunction]
pub fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
    let distance = levenshtein_distance(s1, s2);
    let max_len = s1.chars().count().max(s2.chars().count());

    if max_len == 0 {
        1.0
    } else {
        1.0 - (distance as f64 / max_len as f64)
    }
}

/// match function equivalent
fn levenshtein_match_internal(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
) -> Vec<(String, f64)> {
    let (actual_min, actual_max) = if min <= max { (min, max) } else { (max, min) };
    let mut matches = Vec::with_capacity(strings.len());

    for s in strings {
        let score = levenshtein_similarity(pattern, &s);
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
    Ok(levenshtein_match_internal(
        pattern, strings, min, max, limit,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("你好", "你好"), 0);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn test_similarity() {
        assert_eq!(levenshtein_similarity("", ""), 1.0);
        assert_eq!(levenshtein_similarity("kitten", "sitting"), 1.0 - 3.0 / 7.0);
        assert_eq!(levenshtein_similarity("abc", "xyz"), 0.0);
    }

    #[test]
    fn test_match() {
        let strings = vec![
            "kitten".to_string(),
            "sitting".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        let matches = levenshtein_match_internal("kitten", strings, 0.0, 1.0, 2);

        // Test exact matches including score precision
        assert!((matches[0].1 - 1.0).abs() < f64::EPSILON);
        assert_eq!(matches[0].0, "kitten");

        assert!((matches[1].1 - (1.0 - 3.0 / 7.0)).abs() < f64::EPSILON);
        assert_eq!(matches[1].0, "sitting");
    }
}
