"""Run the saved-model evaluation in a fresh Jupyter kernel."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Also compare results against the included reference CSV.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["JUPYTER_RUNTIME_DIR"] = str(output / ".jupyter")
    env["IPYTHONDIR"] = str(output / ".ipython")
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--execute",
         "--to", "notebook", "--ExecutePreprocessor.timeout=600",
         "--ExecutePreprocessor.kernel_name=python3",
         "--output", "evaluation.ipynb", "--output-dir", str(output),
         str(root / "test.ipynb")],
        cwd=root, env=env, check=True,
    )
    result_path = output / "test_results.csv"
    print(result_path.read_text(encoding="utf-8"))
    if args.check:
        def read_rows(path):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return {(row["test"], row["model"]): row
                        for row in csv.DictReader(handle)}

        expected = read_rows(root / "test_results.csv")
        actual = read_rows(result_path)
        if actual.keys() != expected.keys():
            raise SystemExit("FAIL: evaluation cases differ from the reference.")
        for key in expected:
            for metric in ("sup_norm", "relative_L2"):
                if not math.isclose(float(actual[key][metric]),
                                    float(expected[key][metric]),
                                    rel_tol=1e-5, abs_tol=1e-8):
                    raise SystemExit(f"FAIL: {key} {metric} differs from the reference.")
        print("PASS: all four cases match the reference (rtol=1e-5, atol=1e-8).")
    print(f"Results and executed notebook: {output}")


if __name__ == "__main__":
    main()
