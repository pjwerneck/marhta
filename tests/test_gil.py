import threading
import time

import marhta

TOLERANCE = 1.1


def run_sequential(func, args, n_times=10):
    start = time.perf_counter_ns()
    for _ in range(n_times):
        func(*args)
    return time.perf_counter_ns() - start


def run_parallel(func, args, n_threads=10):
    threads = []
    start = time.perf_counter_ns()

    for _ in range(n_threads):
        t = threading.Thread(target=func, args=args)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return time.perf_counter_ns() - start


# We run the tests in sequence and in parallel to check if the GIL is released.
# If the GIL is released, the parallel tests should be faster than the
# sequential ones. If the GIL is not released, the parallel tests should be
# slower than the sequential ones.


def test_levenshtein_gil_release():
    # Short strings should not release GIL (parallel slower)
    seq_time = run_sequential(marhta.levenshtein_distance, ("short", "string"))
    par_time = run_parallel(marhta.levenshtein_distance, ("short", "string"))
    assert par_time > (seq_time * TOLERANCE), "GIL should not be released for short strings"

    # Long strings should release GIL (parallel faster)
    threshold = 64  # LEVENSHTEIN_GIL_RELEASE_THRESHOLD
    long_a = "a" * (threshold + 1)
    long_b = "b" * (threshold + 1)
    seq_time = run_sequential(marhta.levenshtein_distance, (long_a, long_b))
    par_time = run_parallel(marhta.levenshtein_distance, (long_a, long_b))
    assert (par_time * TOLERANCE) < seq_time, "GIL should be released for long strings"


def test_jaro_winkler_gil_release():
    # Short strings should not release GIL (parallel slower)
    seq_time = run_sequential(marhta.jaro_winkler_similarity, ("short", "string", 0.1, 4))
    par_time = run_parallel(marhta.jaro_winkler_similarity, ("short", "string", 0.1, 4))
    assert par_time > (seq_time * TOLERANCE), "GIL should not be released for short strings"

    # Long strings should release GIL (parallel faster)
    threshold = 128  # JARO_WINKLER_GIL_RELEASE_THRESHOLD
    long_a = "a" * (threshold + 1)
    long_b = "b" * (threshold + 1)
    seq_time = run_sequential(marhta.jaro_winkler_similarity, (long_a, long_b, 0.1, 4))
    par_time = run_parallel(marhta.jaro_winkler_similarity, (long_a, long_b, 0.1, 4))
    assert (par_time * TOLERANCE) < seq_time, "GIL should be released for long strings"
