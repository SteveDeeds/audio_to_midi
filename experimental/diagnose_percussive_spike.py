"""
experimental/diagnose_percussive_spike.py
=========================================
Diagnose why percussive_spike feature is so noisy and inconsistent across models.

The issue: _percussive_spike_at_grid() sums raw linear CQT magnitudes without normalization.
This creates values that vary wildly by:
  1. Overall audio loudness (quiet files → small values, loud files → large values)
  2. Spectral content (files with more energy across all bins → larger spikes)
  3. Accumulation over time/frequency (no per-note or per-window normalization)

Result: Feature is unnormalized, scale-dependent, and therefore noisy.
"""

import json
from pathlib import Path
import numpy as np


def load_profile_candidates(path: Path) -> dict:
    """Extract candidate training data with focus on percussive_spike."""
    data = json.loads(path.read_text(encoding="utf-8"))
    
    candidates = {
        "positive": {"spikes": [], "all_rows": []},
        "negative": {"spikes": [], "all_rows": []},
    }
    
    for item in data.get("cand", []):
        row = item.get("row", [])
        label = item.get("label", 0)
        if len(row) < 4:
            continue
        
        transient, chroma_delta, energy_db, percussive_spike = row[0], row[1], row[2], row[3]
        
        label_key = "positive" if label == 1 else "negative"
        candidates[label_key]["spikes"].append(float(percussive_spike))
        candidates[label_key]["all_rows"].append({
            "transient": float(transient),
            "chroma_delta": float(chroma_delta),
            "energy_db": float(energy_db),
            "percussive_spike": float(percussive_spike),
        })
    
    return candidates


def analyze_percussive_spike(profiles: dict) -> None:
    """Analyze why percussive_spike is problematic."""
    
    print("\n" + "=" * 100)
    print("PERCUSSIVE SPIKE DIAGNOSTIC: Explaining Inconsistency Across Models")
    print("=" * 100)
    
    print("\nFEATURE CALCULATION METHOD:")
    print("""
    percussive_spike = inner - (outer_left + outer_right)
    where:
      inner = sum(cqt_mag[:, center_frames])  
      outer = sum(cqt_mag[:, wing_frames])
    
    PROBLEM: Uses RAW LINEAR CQT magnitudes WITHOUT normalization.
    - cqt_mag = np.abs(librosa.cqt(audio))  [linear, not dB]
    - Sums across ALL frequency bins
    - Result: Massively scale-dependent!
    """)
    
    print("\n" + "-" * 100)
    print("DISTRIBUTION ANALYSIS BY PROFILE")
    print("-" * 100)
    
    for name, cands in profiles.items():
        pos_spikes = np.array(cands["positive"]["spikes"])
        neg_spikes = np.array(cands["negative"]["spikes"])
        
        print(f"\n{name}:")
        print(f"  Positive samples (keep): n={len(pos_spikes)}")
        if len(pos_spikes) > 0:
            print(f"    Values: min={np.min(pos_spikes):10.2f}, median={np.median(pos_spikes):10.2f}, max={np.max(pos_spikes):10.2f}")
            print(f"    Std dev: {np.std(pos_spikes):10.2f}")
            print(f"    Range: {np.max(pos_spikes) - np.min(pos_spikes):10.2f}")
            
            # Count near-zero
            near_zero = np.sum(np.abs(pos_spikes) < 1.0)
            print(f"    Near-zero (<1.0): {near_zero}/{len(pos_spikes)} ({100*near_zero//len(pos_spikes)}%)")
        
        print(f"  Negative samples (drop): n={len(neg_spikes)}")
        if len(neg_spikes) > 0:
            print(f"    Values: min={np.min(neg_spikes):10.2f}, median={np.median(neg_spikes):10.2f}, max={np.max(neg_spikes):10.2f}")
            print(f"    Std dev: {np.std(neg_spikes):10.2f}")
            print(f"    Range: {np.max(neg_spikes) - np.min(neg_spikes):10.2f}")
            
            near_zero = np.sum(np.abs(neg_spikes) < 1.0)
            print(f"    Near-zero (<1.0): {near_zero}/{len(neg_spikes)} ({100*near_zero//len(neg_spikes)}%)")
        
        # Overlap analysis
        if len(pos_spikes) > 0 and len(neg_spikes) > 0:
            pos_median = np.median(pos_spikes)
            neg_median = np.median(neg_spikes)
            overlap = len([x for x in pos_spikes if np.min(neg_spikes) <= x <= np.max(neg_spikes)])
            print(f"  Separation: pos_median={pos_median:.2f}, neg_median={neg_median:.2f}, Δ={pos_median - neg_median:.2f}")
            print(f"  Overlap: {overlap}/{len(pos_spikes)} positive samples in negative range ({100*overlap//len(pos_spikes)}%)")
    
    print("\n" + "-" * 100)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 100)
    print("""
    WHY THE FEATURE IS BROKEN:
    
    1. NO NORMALIZATION BY ENERGY
       - Piano_Chords has all 0.0 values → likely computed on very quiet audio
       - Bass has ±300 values → computed on loudly recorded audio
       - Same musical content, vastly different feature values!
       
    2. SUMS ENTIRE SPECTROGRAM
       - inner = sum across ALL frequency bins (e.g., 84 bins for 7-octave CQT)
       - A strong attack ×84 bins = huge number
       - A quiet note ×84 bins = tiny number
       - No per-frequency weighting or filtering
       
    3. DEPENDS ON UNKNOWN FACTORS
       - Recording level/gain of the original audio
       - Microphone distance/type
       - Instrument tone (bright instruments have broader spectra)
       - Nothing to do with actual musicality!
    
    4. SCALE MISMATCH WITH OTHER FEATURES
       - transient: normalized 0-1 range → stable across files
       - chroma_delta: normalized 0-1 range → stable across files
       - energy_db: dB scale → stable across files
       - percussive_spike: raw sums → UNSTABLE, highly file-dependent!
    """)
    
    print("\n" + "-" * 100)
    print("SOLUTION RECOMMENDATIONS")
    print("-" * 100)
    print("""
    FIX 1: Normalize by window energy (RECOMMENDED)
    ─────────────────────────────────────────────
    Instead of:
      spike = inner - (outer_left + outer_right)
    
    Use normalized ratio:
      total_energy = inner + outer_left + outer_right
      if total_energy > 0:
        spike_ratio = inner / total_energy  # 0-1 range
      else:
        spike_ratio = 0
    
    Benefit: Values become 0-1 range, independent of recording loudness
    
    
    FIX 2: Use log-domain calculation
    ─────────────────────────────────
    Convert cqt_mag to log BEFORE summing:
      log_inner = sum(np.log1p(cqt_mag[:, center]))
      log_outer = sum(np.log1p(cqt_mag[:, wings]))
      spike_log = log_inner - log_outer  # Now dB-like units
    
    Benefit: Stabilizes large dynamic range, more robust to loudness
    
    
    FIX 3: Weight by frequency content
    ──────────────────────────────────
    Only compute spike in mid-high frequencies where transients are:
      transient_bands = cqt_mag[60:84, :]  # Skip low bass
      spike = sum(transient_bands[center]) - sum(transient_bands[wings])
    
    Benefit: Captures percussiveness without bass rumble interference
    
    
    RECOMMENDATION: Apply Fix 1 + Fix 3 together
    ────────────────────────────────────────────
    1. Filter to transient-relevant frequencies (60-84 bins)
    2. Normalize by total energy in window
    3. Result: 0-1 stable feature, instrument-independent
    """)


def main():
    profiles_dir = Path("model_profiles")
    profiles_data = {}
    
    for name in ["bass", "Guitar", "Piano_Chords"]:
        path = profiles_dir / f"{name}.json"
        if path.exists():
            try:
                profiles_data[name] = load_profile_candidates(path)
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    if profiles_data:
        analyze_percussive_spike(profiles_data)
    else:
        print("No profiles found!")


if __name__ == "__main__":
    main()
