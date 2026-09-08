"""The special functions must match SciPy to close to double precision."""

import math

from actstats import special

from reference_values import SPECIAL


def _close(got, expected, rtol=1e-11):
    if math.isinf(expected):
        assert math.isinf(got) and (got > 0) == (expected > 0)
        return
    if expected == 0.0:
        assert abs(got) < 1e-300
        return
    assert abs(got - expected) <= rtol * abs(expected), f"{got!r} != {expected!r}"


def test_gammainc():
    for a, x, expected in SPECIAL["gammainc"]:
        _close(special.gammainc(a, x), expected)


def test_gammaincc():
    for a, x, expected in SPECIAL["gammaincc"]:
        _close(special.gammaincc(a, x), expected)


def test_gammainc_complement():
    for a in (0.2, 1.0, 7.5, 200.0):
        for x in (0.01, 1.0, 7.5, 300.0):
            _close(special.gammainc(a, x) + special.gammaincc(a, x), 1.0, 1e-14)


def test_betainc():
    for a, b, x, expected in SPECIAL["betainc"]:
        _close(special.betainc(a, b, x), expected)


def test_betainc_symmetry():
    for a, b in ((0.3, 2.0), (5.0, 5.0), (30.0, 0.7)):
        for x in (0.01, 0.25, 0.5, 0.9):
            _close(
                special.betainc(a, b, x) + special.betainc(b, a, 1.0 - x), 1.0, 1e-13
            )


def test_ndtr():
    for x, expected in SPECIAL["ndtr"]:
        _close(special.ndtr(x), expected, 1e-13)


def test_ndtri():
    for p, expected in SPECIAL["ndtri"]:
        if expected == 0.0:
            assert special.ndtri(p) == 0.0
        else:
            _close(special.ndtri(p), expected, 1e-12)


def test_ndtri_roundtrip():
    for p in (1e-15, 1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-9):
        _close(special.ndtr(special.ndtri(p)), p, 1e-12)


def test_digamma():
    for x, expected in SPECIAL["digamma"]:
        _close(special.digamma(x), expected, 1e-12)


def test_trigamma():
    for x, expected in SPECIAL["trigamma"]:
        _close(special.trigamma(x), expected, 1e-12)


def test_digamma_recurrence():
    # psi(x + 1) = psi(x) + 1 / x
    for x in (0.2, 1.7, 4.0, 11.9, 300.0):
        _close(special.digamma(x + 1.0), special.digamma(x) + 1.0 / x, 1e-12)


def test_gammaincinv():
    for a, p, expected in SPECIAL["gammaincinv"]:
        _close(special.gammaincinv(a, p), expected, 1e-10)


def test_gammainccinv():
    for a, q, expected in SPECIAL["gammainccinv"]:
        _close(special.gammainccinv(a, q), expected, 1e-10)


def test_betaincinv():
    for a, b, p, expected in SPECIAL["betaincinv"]:
        _close(special.betaincinv(a, b, p), expected, 1e-10)


def test_inverse_roundtrips():
    for a in (0.05, 0.9, 3.0, 60.0):
        for p in (1e-12, 1e-4, 0.3, 0.5, 0.9, 1 - 1e-9):
            _close(special.gammainc(a, special.gammaincinv(a, p)), p, 1e-9)
            q = 1.0 - p
            if q > 0.0:
                _close(special.gammaincc(a, special.gammainccinv(a, q)), q, 1e-9)
    for a, b in ((0.4, 0.6), (2.0, 5.0), (40.0, 3.0)):
        for p in (1e-10, 0.01, 0.5, 0.99):
            _close(special.betainc(a, b, special.betaincinv(a, b, p)), p, 1e-9)


def test_boundaries():
    assert special.gammainc(2.0, 0.0) == 0.0
    assert special.gammaincc(2.0, 0.0) == 1.0
    assert special.gammaincinv(2.0, 0.0) == 0.0
    assert math.isinf(special.gammaincinv(2.0, 1.0))
    assert special.betainc(2.0, 3.0, 0.0) == 0.0
    assert special.betainc(2.0, 3.0, 1.0) == 1.0
    assert special.betaincinv(2.0, 3.0, 0.0) == 0.0
    assert special.betaincinv(2.0, 3.0, 1.0) == 1.0
    assert math.isinf(special.ndtri(1.0)) and special.ndtri(1.0) > 0
    assert math.isinf(special.ndtri(0.0)) and special.ndtri(0.0) < 0


def test_domain_errors():
    for call in (
        lambda: special.gammainc(-1.0, 1.0),
        lambda: special.gammainc(1.0, -1.0),
        lambda: special.betainc(0.0, 1.0, 0.5),
        lambda: special.ndtri(1.5),
        lambda: special.gammaincinv(1.0, 2.0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
