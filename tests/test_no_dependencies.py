"""actstats must not import NumPy or SciPy -- that is the point of the package."""

import subprocess
import sys


def test_import_pulls_in_no_third_party_modules():
    code = (
        "import sys, actstats\n"
        "from actstats import actuarial\n"
        "actuarial.lognormal(0.5, 0.2).rvs(size=10)\n"
        "actuarial.gamma.fit([1.0, 2.0, 3.5, 0.5])\n"
        "actuarial.poisson(3.0).ppf(0.9)\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'numpy', 'scipy', 'pandas'})\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"


def test_only_standard_library_imports_at_module_scope():
    import pathlib
    import re

    allowed = {
        "math",
        "random",
        "datetime",
        "bisect",
        "typing",
        "__future__",
        "actstats",
    }
    root = pathlib.Path(__file__).resolve().parent.parent / "actstats"
    pattern = re.compile(r"^(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
    for path in root.rglob("*.py"):
        for match in pattern.finditer(path.read_text()):
            top = match.group(1).split(".")[0]
            assert top in allowed, f"{path.name} imports {top} at module scope"


def test_numpy_bridge_is_optional_and_lazy():
    # Sample.to_numpy is the single place NumPy is named, and only inside the
    # function body, so it costs nothing unless it is called.
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "actstats" / "sample.py"
    ).read_text()
    for line in source.splitlines():
        if "import numpy" in line:
            assert line.startswith(" "), "the NumPy import must not be at module scope"
            break
    else:
        raise AssertionError("expected the optional NumPy bridge in sample.py")
