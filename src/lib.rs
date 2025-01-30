/// String similarity and fuzzy matching algorithms for Python
/// 
/// This module provides implementations of common string similarity algorithms:
/// 
/// * Levenshtein distance and similarity measures
/// * Jaro-Winkler distance and similarity measures
/// 
/// Each algorithm provides distance, similarity, and fuzzy matching capabilities.
use pyo3::prelude::*;

mod jaro_winkler;
mod levenshtein;

#[pymodule]
fn marhta(_py: Python, m: &PyModule) -> PyResult<()> {
    // Levenshtein functions
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein::levenshtein_match, m)?)?;
    // Jaro-Winkler functions
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler::jaro_winkler_match, m)?)?;

    Ok(())
}
