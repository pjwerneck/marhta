# marhta
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Fast string similarity algorithms

*This library is still in development and may not be stable.*

## Installation

```bash
pip install marhta
```

## Usage

```python
from marhta import levenshtein_similarity, jaro_winkler_similarity

# Calculate string similarities
print(levenshtein_similarity("hello", "helo"))  # 0.8
print(jaro_winkler_similarity("martha", "marhta"))  # 0.961

# Find best matches
from marhta import levenshtein_match
strings = ["apple", "banana", "orange", "pear"]
matches = levenshtein_match("aple", strings)
print(matches)  # [("apple", 0.8), ("pear", 0.5)]
```

## Features

- Levenshtein distance and similarity measures
- Jaro-Winkler distance and similarity measures
- Fuzzy string matching with customizable thresholds

## Performance

Written in Rust for improved performance, while maintaining a Pythonic API.
