use pyo3::prelude::*;
use std::cmp::{max, min};

fn jaro_winkler_match_internal(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
) -> Vec<(String, f64)> {
    let (actual_min, actual_max) = if min <= max { (min, max) } else { (max, min) };
    let mut matches = Vec::with_capacity(strings.len());

    for s in strings {
        let score = jaro_winkler_similarity(pattern, &s);
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
#[pyo3(signature = (pattern, strings, min = 0.0, max = 1.0, limit = 5))]
/// Find the best Jaro-Winkler matches for a pattern in a list of strings
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
pub fn jaro_winkler_match(
    pattern: &str,
    strings: Vec<String>,
    min: f64,
    max: f64,
    limit: usize,
) -> PyResult<Vec<(String, f64)>> {
    Ok(jaro_winkler_match_internal(
        pattern, strings, min, max, limit,
    ))
}

fn matching_characters(a: &str, b: &str, max_distance: usize) -> (usize, usize) {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut matches = 0;
    let mut transpositions = 0;
    let mut b_matches = vec![false; b_chars.len()];
    let mut match_indexes = Vec::new();

    // Find matches
    for (i, &a_char) in a_chars.iter().enumerate() {
        let start = i.saturating_sub(max_distance);
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

    // Count transpositions
    for i in 0..match_indexes.len() {
        for j in i + 1..match_indexes.len() {
            if match_indexes[i].1 > match_indexes[j].1 {
                transpositions += 1;
            }
        }
    }

    (matches, transpositions * 2)
}

#[pyfunction]
/// Calculate the Jaro-Winkler similarity between two strings
/// 
/// The Jaro-Winkler similarity is a measure of string similarity optimized for short strings
/// like names and postcodes. The score ranges from 0.0 (completely different) to 
/// 1.0 (exact match).
/// 
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
/// 
/// Returns:
///     float: Similarity score between 0.0 and 1.0
pub fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let max_distance = (max(s1.chars().count(), s2.chars().count()) / 2).saturating_sub(1);
    let (matches, transpositions) = matching_characters(s1, s2, max_distance);

    if matches == 0 {
        return 0.0;
    }

    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    let s1_len = s1.chars().count() as f64;
    let s2_len = s2.chars().count() as f64;

    let jaro = (m / s1_len + m / s2_len + (m - t) / m) / 3.0;

    // Winkler adjustment
    let l = s1
        .chars()
        .zip(s2.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count() as f64;

    jaro + (l * 0.1 * (1.0 - jaro))
}

#[pyfunction]
/// Calculate the Jaro-Winkler distance between two strings
/// 
/// The Jaro-Winkler distance is the complement of the Jaro-Winkler similarity (1 - similarity).
/// The score ranges from 0.0 (exact match) to 1.0 (completely different).
/// 
/// Args:
///     s1 (str): First string to compare
///     s2 (str): Second string to compare
/// 
/// Returns:
///     float: Distance score between 0.0 and 1.0
pub fn jaro_winkler_distance(s1: &str, s2: &str) -> f64 {
    1.0 - jaro_winkler_similarity(s1, s2)
}



#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_jaro_winkler() {
        // Standard test cases
        assert_relative_eq!(
            jaro_winkler_similarity("MARTHA", "MARHTA"),
            0.961,
            epsilon = 0.001
        );
        assert_relative_eq!(
            jaro_winkler_similarity("DWAYNE", "DUANE"),
            0.840,
            epsilon = 0.001
        );
        assert_relative_eq!(
            jaro_winkler_similarity("ABCD", "EFGH"),
            0.0,
            epsilon = 0.001
        );

        // Edge cases
        assert_eq!(jaro_winkler_similarity("", ""), 1.0);
        assert_eq!(jaro_winkler_similarity("test", ""), 0.0);
        assert_eq!(jaro_winkler_similarity("", "test"), 0.0);
    }

    #[test]
    fn test_jaro_winkler_match() {
        let strings = vec![
            "apple".to_string(),
            "apples".to_string(),
            "aple".to_string(),
            "appliance".to_string(),
        ];

        let result = jaro_winkler_match_internal("apple", strings, 0.0, 1.0, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].0, "apple");
        assert_relative_eq!(result[0].1, 1.0, epsilon = 0.001);
        assert_eq!(result[1].0, "apples");
        assert_relative_eq!(result[1].1, 0.967, epsilon = 0.001);
    }
}
