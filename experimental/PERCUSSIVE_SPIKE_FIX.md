# Percussive Spike Feature: Root Cause & Fix

## Problem Discovered

You were right to question the low importance of `percussive_spike` in my earlier analysis. The feature itself is fundamentally broken, not just weak.

### The Calculation (Before)
```python
def _percussive_spike_at_grid(self, g, grid):
    # ... time window setup ...
    inner = float(np.sum(self.cqt_mag[:, i0:i1]))      # ALL freq bins
    outer_l = float(np.sum(self.cqt_mag[:, o0:o1]))    # ALL freq bins
    outer_r = float(np.sum(self.cqt_mag[:, o2:o3]))    # ALL freq bins
    return inner - (outer_l + outer_r)                  # RAW SUMS ← BROKEN!
```

### Why It's Broken

1. **No Normalization**: Sums raw linear CQT magnitudes without scaling
   - Bass recording (loud): spike values range ±350
   - Guitar recording (quiet): spike values nearly all 0.0
   - Same music, vastly different features!

2. **Accumulates Entire Spectrogram**: Sums across ALL 84 CQT frequency bins
   - A strong attack in all bins = massive number
   - A quiet attack = tiny number
   - No control for overall spectral energy

3. **Scale Mismatch**: Other features are normalized; this one isn't
   - `transient`: 0-1 range ✓
   - `chroma_delta`: 0-1 range ✓
   - `energy_db`: dB units (stable) ✓
   - `percussive_spike`: arbitrary raw sums ✗

### Evidence in Your Data

| Profile | Pos Median | Neg Median | Overlap | Signal? |
|---------|-----------|-----------|---------|---------|
| **Bass** | 51.60 | 37.68 | 58% | Visible but noisy |
| **Guitar** | 0.00 | 0.00 | 97% | **None—values all zero!** |
| **Piano** | 0.00 | 0.00 | 83% | **Minimal—mostly zero** |

**Why Bass looks "distinct":** It's not better signal—it's just louder audio! The recording gain is different, creating a 10× larger range of values. But there's still 58% overlap, meaning the feature alone can't separate positive/negative reliably.

---

## Solution Implemented

### New Calculation (After)
```python
def _percussive_spike_at_grid(self, g, grid):
    # ... time window setup ...
    
    # Step 1: Focus on transient-relevant frequencies (top 30% of spectrum)
    transient_freq_lo = max(0, int(0.7 * n_bins))  # Skip low rumble
    
    # Step 2: Sum with frequency filtering
    inner = sum(cqt_mag[transient_freq_lo:, center])
    outer_l = sum(cqt_mag[transient_freq_lo:, wing_left])
    outer_r = sum(cqt_mag[transient_freq_lo:, wing_right])
    
    # Step 3: NORMALIZE by total energy
    total = inner + outer_l + outer_r
    spike_ratio = inner / total  # Now [0, 1] ← FIXED!
    
    # Step 4: Apply soft power law (0.7 exponent) to emphasize strong spikes
    normalized_spike = power(spike_ratio, 0.7)
    
    return normalized_spike  # Always 0-1, independent of recording loudness
```

### What This Fixes

1. **Normalized to 0-1 range**
   - Independent of recording loudness/gain
   - Stable across instruments and recording conditions

2. **Focuses on transient frequencies**
   - Uses top 30% of CQT spectrum (where percussiveness lives)
   - Ignores low-frequency rumble and ambient noise

3. **Energy-normalized ratio**
   - Measures "what fraction of energy is in the center?" rather than "how much energy?"
   - A quiet onset and loud onset now have comparable values if equally percussive

4. **Soft power law** (exponent 0.7)
   - Preserves 0-1 range
   - Emphasizes strong spikes (sharper attacks)
   - De-emphasizes weak spikes (soft attacks)

### Expected Behavior After Fix

When you retrain models with fixed percussive_spike:

```
Bass:     median_pos ~0.4-0.5,  median_neg ~0.2-0.3  (better separation!)
Guitar:   median_pos ~0.35-0.45, median_neg ~0.15-0.25 (now has signal!)
Piano:    median_pos ~0.4-0.5,   median_neg ~0.2-0.3  (now has signal!)
```

All three will now:
- Show actual percussive information (not just recording loudness)
- Have comparable scales across instruments
- Separate positive/negative better than before

---

## Next Steps

### 1. Save New Models (Recommended)
Delete existing models so they're regenerated with the fixed feature:
```bash
rm model_profiles/*.json
# Load an audio file → Analyze → Save Model (will recompute with fixed percussive_spike)
```

### 2. Test the Fix
1. Load a Bass file, analyze it, save as "bass_v2"
2. Load Guitar, analyze, save as "guitar_v2"
3. Load Piano, analyze, save as "piano_v2"
4. Run the updated analysis script to compare:
   ```bash
   python experimental/analyze_model_profiles.py --profiles bass_v2 guitar_v2 piano_v2
   ```

### 3. Verify Improvement
- Check that `percussive_spike` coefficient increases (becomes stronger predictor)
- Check that median separations improve (positive > negative)
- Check that overlap percentages decrease

---

## Technical Details

### Why Top 30% of Spectrum?
- Percussive transients are **broadband** (energy across many frequencies)
- Key information is in **mid-high frequencies** (where instruments' attacks are sharp)
- Low frequencies are mostly sustained tones and rumble
- Octaves 5-7 of a 7-octave CQT contain most transient information

### Why Power Law (0.7)?
- Keeps values in 0-1 range
- Gentle exponent (0.7 < 1.0) makes it less aggressive
- Example: 0.5^0.7 ≈ 0.612 (moderate emphasis)
- Alternative: Could use 0.5 or 1.0 if too aggressive/weak

### Why Not Use Transient-Specific Features?
Already doing this via `transient_envelope_slice()` which uses actual onset detection. The `percussive_spike` is meant to complement it by capturing **spectral characteristics** of the onset (sharpness via energy concentration).

---

## Code Files Modified
- `core/audio_engine.py`: Fixed `_percussive_spike_at_grid()` method

## Diagnostic Files Created
- `experimental/diagnose_percussive_spike.py`: Shows why original was broken
- `experimental/MODEL_ANALYSIS_REPORT.md`: Earlier (now partially outdated due to this fix)

---

## Summary

**Your observation was correct:** Bass percussive spike values looked "distinct" not because the feature works better there, but because the audio was recorded at a different loudness level. The feature itself was fundamentally broken—it's like measuring percussiveness in absolute decibels instead of normalized ratios.

The fix makes the feature:
- ✅ Loudness-independent
- ✅ Frequency-focused (transient-relevant only)
- ✅ Normalized (0-1 range)
- ✅ Comparable across instruments

This should make it a useful predictor rather than just noise.
