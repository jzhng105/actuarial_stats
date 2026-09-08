"""Benchmark actstats against SciPy/NumPy.

Run it with SciPy installed to get the comparison columns; without SciPy it
still reports the pure-Python timings::

    python benchmarks/benchmark.py

Numbers are per-call for the scalar rows and per-batch for the sampling rows.
Pure Python wins where per-call overhead dominates (scalar evaluation, small
batches, event-by-event simulation) and loses to vectorised NumPy on large
batches -- the trade-off is discussed in the README.
"""

from __future__ import annotations

import math
import sys
import time
from timeit import Timer

sys.path.insert(0, ".")

import actstats
from actstats import actuarial

try:
    import numpy as np
    import scipy.stats as st

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - the point of the package
    HAVE_SCIPY = False


def bench(func, min_time=0.35):
    """Return seconds per call, auto-scaling the repeat count."""
    timer = Timer(func)
    n = 1
    while True:
        elapsed = timer.timeit(n)
        if elapsed >= min_time or n > 10_000_000:
            return elapsed / n
        n = max(int(n * max(2.0, min_time / max(elapsed, 1e-9))), n * 2)


def report(rows):
    width = max(len(r[0]) for r in rows) + 2
    print(f"{'benchmark':{width}s}{'actstats':>13s}{'scipy/numpy':>14s}{'speedup':>10s}")
    print("-" * (width + 37))
    for label, ours, theirs in rows:
        if theirs is None:
            print(f"{label:{width}s}{_fmt(ours):>13s}{'-':>14s}{'-':>10s}")
        else:
            ratio = theirs / ours
            mark = f"{ratio:.1f}x" if ratio >= 1.0 else f"{1/ratio:.1f}x slower"
            print(f"{label:{width}s}{_fmt(ours):>13s}{_fmt(theirs):>14s}{mark:>10s}")


def _fmt(seconds):
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e9:.1f} ns"


def main():
    actstats.seed(1)
    rows = []

    ln = actuarial.lognormal(9.0, 1.3)
    ga = actuarial.gamma(2.5, 4000.0)
    po = actuarial.poisson(3.7)
    if HAVE_SCIPY:
        sp_ln = st.lognorm(1.3, 0, math.exp(9.0))
        sp_ga = st.gamma(2.5, 0, 4000.0)
        sp_po = st.poisson(3.7)

    # --- scalar evaluation ------------------------------------------------
    rows.append((
        "lognormal.pdf(x)  scalar",
        bench(lambda: ln.pdf(12000.0)),
        bench(lambda: sp_ln.pdf(12000.0)) if HAVE_SCIPY else None,
    ))
    rows.append((
        "lognormal.cdf(x)  scalar",
        bench(lambda: ln.cdf(12000.0)),
        bench(lambda: sp_ln.cdf(12000.0)) if HAVE_SCIPY else None,
    ))
    rows.append((
        "gamma.ppf(0.995)  scalar",
        bench(lambda: ga.ppf(0.995)),
        bench(lambda: sp_ga.ppf(0.995)) if HAVE_SCIPY else None,
    ))
    rows.append((
        "poisson.pmf(4)    scalar",
        bench(lambda: po.pmf(4)),
        bench(lambda: sp_po.pmf(4)) if HAVE_SCIPY else None,
    ))

    # --- sampling ---------------------------------------------------------
    for size in (1, 10, 100, 1000, 100000):
        rows.append((
            f"lognormal.rvs(size={size})",
            bench(lambda s=size: ln.rvs(size=s)),
            bench(lambda s=size: sp_ln.rvs(size=s)) if HAVE_SCIPY else None,
        ))
    for size in (1, 1000, 100000):
        rows.append((
            f"gamma.rvs(size={size})",
            bench(lambda s=size: ga.rvs(size=s)),
            bench(lambda s=size: sp_ga.rvs(size=s)) if HAVE_SCIPY else None,
        ))
        rows.append((
            f"poisson.rvs(size={size})",
            bench(lambda s=size: po.rvs(size=s)),
            bench(lambda s=size: sp_po.rvs(size=s)) if HAVE_SCIPY else None,
        ))

    # --- a realistic collective risk model --------------------------------
    def aggregate_actstats(trials=10000):
        """Frequency-severity aggregate loss, one trial at a time."""
        freq = actuarial.poisson(3.7)
        sev = actuarial.lognormal(9.0, 1.3)
        total = 0.0
        for _ in range(trials):
            n = freq.rvs()
            for _ in range(n):
                total += sev.rvs()
        return total

    def aggregate_scipy(trials=10000):
        freq = st.poisson(3.7)
        sev = st.lognorm(1.3, 0, math.exp(9.0))
        total = 0.0
        for _ in range(trials):
            n = freq.rvs()
            if n:
                total += sev.rvs(size=n).sum()
        return total

    def aggregate_numpy(trials=10000):
        counts = np.random.poisson(3.7, trials)
        return np.random.lognormal(9.0, 1.3, counts.sum()).sum()

    rows.append((
        "aggregate loss, 10k trials (loop)",
        bench(aggregate_actstats, min_time=1.0),
        bench(aggregate_scipy, min_time=1.0) if HAVE_SCIPY else None,
    ))
    if HAVE_SCIPY:
        rows.append((
            "aggregate loss, 10k trials (vectorised numpy)",
            bench(aggregate_actstats, min_time=1.0),
            bench(aggregate_numpy, min_time=1.0),
        ))

    # --- fitting ----------------------------------------------------------
    sample = actuarial.gamma(2.5, 4000.0).rvs(size=5000)
    arr = list(sample)
    rows.append((
        "gamma.fit(5000 obs)",
        bench(lambda: actuarial.gamma.fit(arr), min_time=1.0),
        bench(lambda: st.gamma.fit(np.asarray(arr), floc=0), min_time=1.0) if HAVE_SCIPY else None,
    ))

    report(rows)

    # --- import cost ------------------------------------------------------
    print("\nimport cost (cold interpreter, best of 3):")
    import subprocess

    def import_time(statement):
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            subprocess.run([sys.executable, "-c", statement], check=True)
            best = min(best, time.perf_counter() - start)
        return best

    base = import_time("pass")
    ours = import_time("import actstats") - base
    print(f"  import actstats            {ours * 1e3:7.1f} ms")
    if HAVE_SCIPY:
        theirs = import_time("import numpy, scipy.stats") - base
        print(f"  import numpy, scipy.stats  {theirs * 1e3:7.1f} ms   ({theirs / ours:.1f}x)")


if __name__ == "__main__":
    main()
