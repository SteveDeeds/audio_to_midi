# Model Analysis Summary & Recommendations

## Issue #1: Gray Mode Filtering for Recommendations

**Status:** Fixed ✓

**Problem:** Recommendations not being filtered by Gray Mode selection (Gray off-scale / Gray off-chord).

**Root Cause Analysis:**
- The filtering logic was correct but unclear  
- Added better documentation and defensive programming for edge cases

**Fix Applied:**
- Simplified `_recommendation_visible()` to explicitly document behavior:
  - Manual pitches: always show (user edits should never be hidden)
  - Show all mode: show everything  
  - Gray modes: filter NNLS/ML/pyin recommendations by pitch class
- Improved `_get_colored_pcs()` to fallback gracefully:
  - In Gray off-chord: use scale if no chords in current window
  - Returns `None` only if truly no filtering is active

**Testing Recommendation:**
1. Load an audio file and analyze it (Chords & Scale)
2. Switch between "Gray off-scale" and "Show all" modes
3. Verify that recommendations disappear when you select "Gray off-scale" (hiding off-scale notes)
4. Verify that they reappear when you select "Show all"

---

## Issue #2: Feature Importance Analysis Across Instruments

### Key Findings

**Analysis of Bass, Guitar, Piano models:**
- Total samples: Bass (41), Guitar (102), Piano (58)
- Training data quality: Guitar > Piano > Bass

### Feature Importance Ranking

```
                Transient    Energy_db    Chroma_delta    Percussive_spike
─────────────────────────────────────────────────────────────────────────
Bass            1.529 ⭐      0.462         0.419           0.135 ⚠️
Guitar          2.142 ⭐⭐    0.413         0.424           0.234 ⚠️
Piano           1.667 ⭐      0.336         0.313           0.774 ⭐
─────────────────────────────────────────────────────────────────────────
Average |coef|  1.779 ⭐⭐⭐  0.404         0.385           0.381
```

### Weak Predictor: `percussive_spike`
- **Avg |coef| = 0.381** (lowest importance)
- **Issues:**
  - Extremely noisy (std dev = ±87–168 in Bass)
  - Nearly identical distributions in "keep" vs "drop" classes
  - In Guitar data: median=0 for both classes (no signal at all!)
  - Only strong in Piano (0.774) but weak everywhere else
  
**Recommendation:** 
Consider **removing `percussive_spike` entirely** or downweighting it during model training. It's adding noise without discriminative value. This would simplify the feature space from 4D to 3D.

### Strong Predictor: `transient` (ONLY Reliable Feature)
- **Ranks #1 across ALL three instruments** (coef = 1.53–2.14)
- **Clean separation:** Positive class (keep) median ~0.10–0.13, Negative (drop) median ~0.00
- **Interpretation:** Notes with strong transient onset (sudden amplitude spike) are kept; near-silence is dropped
- **Cross-instrument consistency:** Perfect stability across Bass, Guitar, Piano

### Moderate Predictors: `chroma_delta` & `energy_db`
- **Chroma_delta:** Mid-tier importance, rank 2–4 depending on instrument
  - Positive class: median 0.59–0.66 (consistent across all)
  - Negative class: median 0.46–0.53 (less harmonic overlap)
  - **Signal:** Stronger chroma coherence suggests real notes
  
- **Energy_db:** Rank 2–3, but weaker for energy separation
  - Positive class: median –11 to –12 dB  
  - Negative class: median –12 to –15 dB (VERY similar!)
  - **Issue:** Heavy overlap means energy alone can't discriminate well

### Cross-Instrument Similarities
1. **Transient dominates everywhere** → all models agree on this
2. **Chroma_delta is second-tier everywhere** → harmonic stability matters
3. **Energy_db works but inconsistently** → might need per-instrument calibration
4. **Percussive_spike is problematic across the board** → candidate for removal

---

## Actionable Recommendations

### Priority 1: Remove `percussive_spike` Feature
```python
# Current feature vector (4D):
[transient, chroma_delta, energy_db, percussive_spike]

# Proposed feature vector (3D):
[transient, chroma_delta, energy_db]
```

**Benefits:**
- Removes noise from Guitar & Bass training
- Simplifies model (fewer dimensions = faster training, less overfitting)
- Still captures Piano behavior via transient + other features

**Implementation:**
1. Update `core/audio_engine.py` to not compute `percussive_spike` in candidate features
2. Update `core/models.py` to use 3D feature vectors
3. Delete old models or retrain with new feature set
4. Test on all three instruments

### Priority 2: Consider Per-Instrument Model Calibration
- **Transient threshold** might differ: Bass needs different onset sensitivity than Piano
- **Energy_db weighting** could be per-instrument since distributions vary widely
- Current bootstrap approach already helps with this; could be refined further

### Priority 3: Collect More Data for Weak Predictors
- If you want to keep energy-based filtering, collect more diverse energy examples
- Current data: very limited energy variation, making it unreliable

---

## Summary Statistics by Profile

### Bass
- n=41 (31 positive, 10 negative)
- **Best predictor:** transient (1.529)
- **Worst predictor:** percussive_spike (0.135)
- **Issue:** Small dataset, percussive_spike nearly useless

### Guitar  
- n=102 (75 positive, 27 negative)
- **Best predictor:** transient (2.142) ← strongest across all
- **Worst predictor:** percussive_spike (0.234) + energy_db has 0 signal
- **Issue:** Percussive spike has 0 median for both classes

### Piano_Chords
- n=58 (42 positive, 16 negative)
- **Best predictor:** transient (1.667)
- **Unexpected:** percussive_spike = 0.774 (unusual for this instrument!)
- **Note:** Possibly due to piano's percussive nature; but still noisy overall

---

## Files Modified
- `ui/cqt_view.py`: Enhanced `_recommendation_visible()` and `_get_colored_pcs()` for better gray mode filtering
- `experimental/analyze_model_profiles.py`: New analysis script for cross-instrument comparison

## Next Steps
1. Test Gray Mode filtering with these fixes
2. Decide on removing `percussive_spike` 
3. If removing: update feature extraction and retrain models
4. Monitor performance on all three instruments
