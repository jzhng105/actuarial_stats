"""The Sample container."""

import math

from actstats import Sample


def test_summary_statistics():
    sample = Sample([1.0, 2.0, 3.0, 4.0])
    assert sample.mean() == 2.5
    assert sample.sum() == 10.0
    assert sample.var() == 1.25
    assert sample.var(ddof=1) == 5.0 / 3.0
    assert abs(sample.std() - math.sqrt(1.25)) < 1e-15
    assert sample.min() == 1.0 and sample.max() == 4.0


def test_quantiles_interpolate_linearly():
    sample = Sample([1.0, 2.0, 3.0, 4.0])
    assert sample.quantile(0.0) == 1.0
    assert sample.quantile(1.0) == 4.0
    assert sample.median() == 2.5
    assert sample.quantile(0.25) == 1.75
    assert list(sample.percentile([0, 100])) == [1.0, 4.0]


def test_arithmetic_is_elementwise():
    sample = Sample([1.0, 2.0, 3.0])
    assert list(sample * 2) == [2.0, 4.0, 6.0]
    assert list(sample + 1) == [2.0, 3.0, 4.0]
    assert list(sample - [1.0, 1.0, 1.0]) == [0.0, 1.0, 2.0]
    assert list(sample / 2) == [0.5, 1.0, 1.5]
    assert list(sample ** 2) == [1.0, 4.0, 9.0]
    assert list(-sample) == [-1.0, -2.0, -3.0]
    assert list(10 - sample) == [9.0, 8.0, 7.0]


def test_mismatched_lengths_raise():
    try:
        Sample([1.0, 2.0]) + [1.0]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_empty_sample_errors():
    for call in (lambda: Sample().mean(), lambda: Sample().quantile(0.5)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_ecdf_and_repr():
    sample = Sample(range(10))
    assert sample.ecdf(5) == 0.5
    assert "n=10" in repr(sample)
    assert repr(Sample([1, 2])) == "Sample([1, 2])"


def test_still_a_list():
    sample = Sample([3.0, 1.0, 2.0])
    assert sorted(sample) == [1.0, 2.0, 3.0]
    assert sample[0] == 3.0
    assert len(sample) == 3
    assert list(iter(sample)) == [3.0, 1.0, 2.0]
