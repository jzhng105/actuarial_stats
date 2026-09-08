"""Run the test suite without any third-party test runner.

``python tests/run_tests.py`` discovers every ``test_*.py`` module beside this
file and calls each ``test_*`` function in it.  The same files also run under
pytest if it happens to be installed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main(argv):
    selection = argv[1:] if len(argv) > 1 else None
    passed = failed = 0
    failures = []
    started = time.perf_counter()
    for path in sorted(HERE.glob("test_*.py")):
        module = load(path)
        names = [n for n in sorted(vars(module)) if n.startswith("test_")]
        for name in names:
            if selection and not any(token in f"{path.stem}.{name}" for token in selection):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            try:
                func()
            except Exception:
                failed += 1
                failures.append((f"{path.stem}.{name}", traceback.format_exc()))
                print("F", end="", flush=True)
            else:
                passed += 1
                print(".", end="", flush=True)
    elapsed = time.perf_counter() - started
    print()
    for name, tb in failures:
        print(f"\n=== FAIL: {name} ===\n{tb}")
    print(f"\n{passed} passed, {failed} failed in {elapsed:.2f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
