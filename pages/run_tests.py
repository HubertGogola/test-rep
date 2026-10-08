"""
Minimal test runner with no dependency on pytest being installed.

Usage:
    python3 tests/run_tests.py

If pytest is available, `python3 -m pytest tests/` also works directly,
since every test file here uses plain pytest-style `test_*` functions.
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_MODULES = [
    "tests.test_pii_formula",
    "tests.test_context_standardization",
    "tests.test_gmm_and_hmm",
    "tests.test_survival_analysis",
    "tests.test_privacy",
    "tests.test_comparison_options",
]


def main():
    total, failed = 0, 0
    for mod_name in TEST_MODULES:
        module = importlib.import_module(mod_name)
        test_funcs = [getattr(module, n) for n in dir(module) if n.startswith("test_")]
        for fn in test_funcs:
            total += 1
            try:
                fn()
                print(f"  PASS  {mod_name}.{fn.__name__}")
            except Exception:
                failed += 1
                print(f"  FAIL  {mod_name}.{fn.__name__}")
                traceback.print_exc()
    print()
    print(f"{total - failed}/{total} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
