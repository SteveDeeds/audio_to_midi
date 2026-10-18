"""
experimental/visualize_training_data.py
======================================
Visualize saved training data from model profiles and derive better bootstrap
candidate examples.

Outputs:
- candidate_pca.png: 2D projection of candidate boundary data (note vs non-note)
- pitch_pca.png: 2D projection of pitch fingerprint vectors
- bootstrap_candidates.json: optional generated bootstrap rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _load_profiles(profiles_dir: Path) -> list[dict]:
    profiles = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_profile_name"] = path.stem
            profiles.append(payload)
        except Exception as exc:
            print(f"[warn] skipping {path.name}: {exc}")
    return profiles


def _collect_candidate_rows(profiles: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    labels = []
    names = []
    for profile in profiles:
        pname = profile.get("_profile_name", "unknown")
        if "cand" in profile:
            for item in profile.get("cand", []):
                row = item.get("row")
                if not isinstance(row, list) or len(row) < 4:
                    continue
                rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                labels.append(int(item.get("label", 0)))
                names.append(str(pname))
        else:
            # Backward compatibility with older schema.
            for row, label in zip(profile.get("cand_X", []), profile.get("cand_y", [])):
                if not isinstance(row, list) or len(row) < 3:
                    continue
                vals = [float(v) for v in row]
                if len(vals) >= 5:
                    mapped = [vals[0], vals[1], vals[2], vals[4]]
                elif len(vals) == 4:
                    mapped = [vals[0], vals[1], vals[2], 0.0]
                else:
                    while len(vals) < 4:
                        vals.append(0.0)
                    mapped = vals[:4]
                rows.append(mapped)
                labels.append(int(label))
                names.append(str(pname))
    if not rows:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int), []
    return np.array(rows, dtype=float), np.array(labels, dtype=int), names


def _collect_pitch_rows(profiles: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vecs = []
    labels = []
    names = []
    for profile in profiles:
        pname = profile.get("_profile_name", "unknown")
        for item in profile.get("pitch", []):
            vec = item.get("vec")
            if not isinstance(vec, list) or not vec:
                continue
            notes = item.get("notes")
            if isinstance(notes, list) and notes:
                # Label by first note pitch class for color grouping.
                label = int(notes[0]) % 12
            else:
                label = -1
            vecs.append([float(v) for v in vec])
            labels.append(label)
            names.append(str(pname))
    if not vecs:
        return np.zeros((0, 1), dtype=float), np.zeros((0,), dtype=int), []
    return np.array(vecs, dtype=float), np.array(labels, dtype=int), names


def _plot_candidate_pca(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xs, y)

    # Decision boundary in PCA space via inverse transform.
    x_min, x_max = Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5
    y_min, y_max = Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    grid_pca = np.column_stack([gx.ravel(), gy.ravel()])
    grid_x = pca.inverse_transform(grid_pca)
    probs = clf.predict_proba(grid_x)[:, 1].reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(gx, gy, probs, levels=20, alpha=0.25, cmap="RdYlBu")
    fig.colorbar(contour, ax=ax, label="P(is note boundary)")

    pos = y == 1
    neg = y == 0
    ax.scatter(Z[neg, 0], Z[neg, 1], c="#1f77b4", label="Negative (non-boundary)", s=34, alpha=0.85)
    ax.scatter(Z[pos, 0], Z[pos, 1], c="#d62728", label="Positive (note boundary)", s=34, alpha=0.85)
    ax.set_title("Candidate Training Data (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_pitch_pca(X: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(9, 7))
    valid = labels >= 0
    if np.any(valid):
        sc = ax.scatter(
            Z[valid, 0], Z[valid, 1], c=labels[valid], cmap="tab20", s=30, alpha=0.85
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Pitch class of first note")
    unknown = ~valid
    if np.any(unknown):
        ax.scatter(Z[unknown, 0], Z[unknown, 1], c="#999999", s=28, alpha=0.7, label="Unknown")

    ax.set_title("Pitch Fingerprint Training Data (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    if np.any(unknown):
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _prototype_rows(X: np.ndarray, y: np.ndarray, per_class: int = 2) -> list[dict]:
    rows = []
    for label in [0, 1]:
        Xi = X[y == label]
        if len(Xi) == 0:
            continue
        k = min(per_class, len(Xi))
        if k == 1:
            centers = np.array([np.median(Xi, axis=0)])
        else:
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            km.fit(Xi)
            centers = km.cluster_centers_
        for c in centers:
            rows.append(
                {
                    "row": [float(c[0]), float(c[1]), float(c[2]), float(c[3])],
                    "label": int(label),
                    "key": None,
                    "source": "bootstrap",
                }
            )
    return rows


def _print_candidate_stats(X: np.ndarray, y: np.ndarray) -> None:
    print("\nCandidate feature stats by class:")
    names = ["transient", "chroma_delta", "energy_db", "percussive_spike"]
    for label, title in [(1, "positive"), (0, "negative")]:
        Xi = X[y == label]
        if len(Xi) == 0:
            continue
        med = np.median(Xi, axis=0)
        q1 = np.percentile(Xi, 25, axis=0)
        q3 = np.percentile(Xi, 75, axis=0)
        print(f"  {title} n={len(Xi)}")
        for i, nm in enumerate(names):
            print(f"    {nm:16s} med={med[i]:8.4f}  iqr=[{q1[i]:8.4f}, {q3[i]:8.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and analyze model profile training data")
    parser.add_argument("--profiles-dir", default="model_profiles", help="Directory containing saved model profiles")
    parser.add_argument("--save-dir", default="experimental/out", help="Directory to save plots and outputs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--generate-bootstrap",
        action="store_true",
        help="Generate bootstrap_candidates.json from profile data",
    )
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    profiles = _load_profiles(profiles_dir)
    if not profiles:
        raise SystemExit(f"No profiles found in {profiles_dir}")

    print(f"Loaded {len(profiles)} profile(s) from {profiles_dir}")

    Xc, yc, _ = _collect_candidate_rows(profiles)
    if len(Xc) >= 4 and len(np.unique(yc)) >= 2:
        _plot_candidate_pca(Xc, yc, save_dir / "candidate_pca.png")
        _print_candidate_stats(Xc, yc)
        print(f"Saved {save_dir / 'candidate_pca.png'}")
    else:
        print("Not enough candidate rows to build candidate PCA plot.")

    Xp, yp, _ = _collect_pitch_rows(profiles)
    if len(Xp) >= 2:
        _plot_pitch_pca(Xp, yp, save_dir / "pitch_pca.png")
        print(f"Saved {save_dir / 'pitch_pca.png'}")
    else:
        print("Not enough pitch rows to build pitch PCA plot.")

    if args.generate_bootstrap and len(Xc) >= 4 and len(np.unique(yc)) >= 2:
        rows = _prototype_rows(Xc, yc, per_class=2)
        payload = {
            "schema": 7,
            "name": "generated_bootstrap",
            "cand": rows,
            "pitch": [],
        }
        out_json = save_dir / "bootstrap_candidates.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {out_json}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
