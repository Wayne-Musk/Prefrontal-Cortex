"""零依赖测试运行器（没装 pytest 也能跑）。

    python tests/run_tests.py
    pytest tests -q          # 装了 pytest 也一样能跑

这里的断言针对的是架构声称的能力，而不只是「代码不崩」。
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    mod_names = sorted(p.stem for p in Path(__file__).parent.glob("test_*.py"))
    passed = failed = 0
    failures = []
    for name in mod_names:
        mod = importlib.import_module(f"tests.{name}")
        tests = [(n, getattr(mod, n)) for n in dir(mod)
                 if n.startswith("test_") and callable(getattr(mod, n))]
        print(f"\n── {name} ({len(tests)} 项) " + "─" * 40)
        for tname, fn in tests:
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS  {tname:<52s} {time.time()-t0:5.2f}s")
            except Exception:
                failed += 1
                failures.append((tname, traceback.format_exc()))
                print(f"  FAIL  {tname:<52s} {time.time()-t0:5.2f}s")
    if failures:
        print("\n" + "=" * 70)
        for tname, tb in failures:
            print(f"\n### {tname}\n{tb}")
    print("\n" + "=" * 70)
    print(f"通过 {passed} / 失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
