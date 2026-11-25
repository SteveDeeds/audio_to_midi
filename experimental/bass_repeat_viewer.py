"""Build a standalone bass repeat visualizer from the stem similarity report."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_REPORT = Path("experimental") / "outputs" / "stem_similarity"


def build_viewer(report: Path) -> Path:
    metadata = json.loads((report / "metadata.json").read_text(encoding="utf-8"))
    matrices = {}
    for kind in ("cqt", "mel"):
        with np.load(report / f"bass-{kind}.npz") as arrays:
            matrix = arrays["one_bar"]
            if matrix.ndim != 2 or not len(matrix) or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Expected a nonempty square bass {kind.upper()} matrix.")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"Nonfinite scores in bass {kind.upper()} matrix.")
            matrices[kind] = matrix.tolist()
    if len(matrices["cqt"]) != len(matrices["mel"]):
        raise ValueError("Bass CQT and Mel matrices must cover the same bars.")
    slices_per_bar = metadata["slices_per_bar"]
    slice_count = slices_per_bar * len(matrices["cqt"])
    with np.load(report / "bass-cqt.npz") as arrays:
        features = arrays["features"]
        if features.ndim != 2 or features.shape[0] < slice_count or features.shape[1] < 36:
            raise ValueError("Bass CQT must cover the complete bars and three octaves in the overlay.")
        cqt = features[:slice_count, :36]
        if not np.all(np.isfinite(cqt)) or np.any(cqt < 0):
            raise ValueError("Invalid bass CQT magnitudes in overlay.")
    data = {
        "firstBar": metadata["first_bar"], "matrices": matrices,
        "slicesPerBar": slices_per_bar, "cqt": cqt.tolist(),
    }
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    output = report / "bass-repeats.html"
    output.write_text(template.replace("__REPORT_DATA__", json.dumps(data)), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(build_viewer(args.report).resolve())


if __name__ == "__main__":
    main()
