"""Active BabySlakh stems for isolated pitch-model comparisons."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    REPO_ROOT
    / "experimental"
    / "datasets"
    / "babyslakh"
    / "babyslakh_16k"
)

CANDIDATES = (
    ("track00013_s07", "monophonic", "Track00013", "S07", "Flute", 1, "standard"),
    ("track00014_s00", "monophonic", "Track00014", "S00", "Trumpet", 1, "standard"),
    ("track00014_s01", "monophonic", "Track00014", "S01", "Electric Bass", 1, "bass"),
    ("track00002_s06", "polyphonic", "Track00002", "S06", "Clavinet", 4, "standard"),
    ("track00014_s05", "polyphonic", "Track00014", "S05", "Electric Piano", 7, "standard"),
    ("track00014_s06", "polyphonic", "Track00014", "S06", "Choir", 4, "standard"),
    ("track00016_s04", "polyphonic", "Track00016", "S04", "String Ensemble", 8, "standard"),
    ("track00016_s10", "polyphonic", "Track00016", "S10", "Acoustic Piano", 3, "standard"),
)