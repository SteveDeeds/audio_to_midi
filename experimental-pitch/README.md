# experimental-pitch

Self-contained bake-off of open-source pitch / note transcription techniques on
Demucs/UVR stems. Nothing here is imported by the main app; it has its own
`.venv` (Python 3.11, CPU torch) so the app's environment is untouched.

## Methods compared

| key | source | type | notes |
|---|---|---|---|
| `nnls_baseline` | this repo's `core/polyphonic_nnls.py` (imported read-only) | 5-harmonic NNLS on CQT | approximates the app's *untrained* path, no user learning |
| `pyin` | librosa (Mauch & Dixon 2014) | probabilistic YIN + HMM | classic DSP mono f0 |
| `crepe_tiny` / `crepe_full` | [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) | CNN on raw audio, Viterbi decoding | mono f0; tiny ≈ 0.5M params, full ≈ 22M |
| `pesto` | [SonyCSLParis/pesto](https://github.com/SonyCSLParis/pesto) | self-supervised, transposition-equivariant CQT net | mono f0, ~130k params |
| `basic_pitch` | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | small CNN on HCQT, onset+note+contour heads | polyphonic, ~36k params, ONNX |
| `piano_hr` | [bytedance/piano_transcription](https://github.com/bytedance/piano_transcription) | Onsets-and-Frames-style CRNN with regressed onset/offset | polyphonic piano, ~43M params |
| `*+onset` | this folder (`common.onset_segment`) | SuperFlux attacks + stable-pitch assignment | re-segments any cached mono f0 contour |
| `*+key` / `*+onset+key` | this folder (`common.key_clean_contour`) | key-aware folding of short non-diatonic runs into bends/slides | bass and vocals |

Mono f0 trackers (pyin/crepe/pesto) are turned into notes by a shared
segmenter (`common.contour_to_notes`): median smoothing, semitone quantisation,
then splitting at spectral-flux onsets so repeated same-pitch notes separate.

## Run

```powershell
cd experimental-pitch
.\setup_env.ps1                                   # once
.venv\Scripts\python.exe run_methods.py           # Firm Foundation, all pitched stems
.venv\Scripts\python.exe resegment.py             # bass '+onset', '+onset+key' and vocals '+key' variants
                                                  # (--key "Bb major" to override the estimate, --no-harmony)
.venv\Scripts\python.exe evaluate.py              # metrics + piano rolls + A/B audio
start "outputs\Firm Foundation (He Won't)\report.html"
```

Other songs, e.g. EchoGrid output (`bass.wav`, `piano.wav`, `guitar.wav`, ... and
`NNNbpm` in the name):

```powershell
.venv\Scripts\python.exe run_methods.py --song-dir <stems folder> [--bpm 111]
.venv\Scripts\python.exe evaluate.py --song-dir <stems folder>
```

Use `--stems`, `--methods`, `--start/--duration` for quick excerpts and
`--force` to recompute cached results. Outputs land in `outputs/<song>/`
(git-ignored): full-song `.mid` per stem×method, `metrics.json`, `report.html`.

Six-stem Demucs separation (htdemucs_6s adds guitar and piano) and a bake-off on
each stem's loudest window, with one output folder per stem:

```powershell
uv pip install --python .venv\Scripts\python.exe demucs soundfile   # once
.venv\Scripts\python.exe separate_6s.py "<mix>.wav"       # -> <mix folder>\htdemucs_6s\<stem>.wav
.venv\Scripts\python.exe run_loudest.py --song-dir "<mix folder>\htdemucs_6s" --bpm 75 [--duration 30]
```

## Evaluation (no ground truth)

`evaluate.py` reports proxy metrics; the A/B audio (left = stem,
right = transcription synth) and piano rolls over the CQT are the real judge.

- **pitch support**: note-frames whose harmonic-sum salience is ≥35% of the frame max (precision proxy)
- **chroma cos**: transcription pitch classes vs. the stem's CQT chroma
- **coverage / ghost**: audible frames with a note / note frames in silence
- **fragments**: notes < 100 ms
- **out-of-key time**: share of note time whose pitch class is outside the song's scale
- **onsets / grid lock**: circular concentration of note-onset phase on the grid (0 = random; drums give the ceiling), latency-invariant; grid is 16th or 16th-triplet, whichever fits the drums
- **agreement**: pairwise onset+pitch F1 between methods, exact vs. any-octave (gap = octave confusion)
- **CPU x realtime / params**: feasibility for inference and on-device fine-tuning

## Findings — Firm Foundation (382 s, 75 BPM, CPU only)

The song has a triplet feel: drum onsets fit a 16th-triplet grid (grid lock 0.72) far
better than straight 16ths.

**Bass** (assumed monophonic)

| method | pitch support | chroma cos | fragments | grid lock | CPU x RT | params |
|---|---|---|---|---|---|---|
| nnls_baseline (app today) | 89.7 | 0.84 | 27% | 0.08 | 90 | 0 |
| pyin | 88.4 | 0.82 | 29% | 0.08 | 2.3 | 0 |
| **pyin+onset** | 86.8 | 0.81 | 0% | **0.31** | 2.3 | 0 |
| crepe_tiny | 92.8 | 0.73 | 20% | 0.14 | 8.3 | 0.49M |
| **crepe_tiny+onset** | 88.9 | 0.75 | 1.5% | **0.32** | 7.9 | 0.49M |
| crepe_full | 92.5 | 0.73 | 20% | 0.15 | 0.4 | 22M |
| pesto | 76.4 | 0.74 | 29% | 0.05 | 17 | 0.13M |
| basic_pitch (mono-reduced) | 85.7 | 0.78 | 4% | 0.10 | 45 | 36k |

- Pitch is largely a solved problem on a clean bass stem: pYIN, CREPE and even the app's
  NNLS agree on 75–86% of notes (onset+pitch F1) with almost no octave confusion.
- **The weak point is note segmentation, not pitch.** Splitting notes where the f0 changes
  turns every slide into a burst of off-grid fragments. Segmenting at attacks and giving each
  segment its stable pitch cut notes from ~1650 to ~500 (≈4 per bar) and quadrupled grid lock.
- CREPE-full is no better than CREPE-tiny and runs at 0.4x realtime — not viable here.
- PESTO's voicing confidence is ~0 on bass (trained on voice); its pitch is the least supported.
- Basic Pitch on bass misses slides, has the lowest agreement, and 17% of its note time
  falls where the stem is silent.

**Vocals**: methods agree less (F1 0.33–0.6). CREPE-tiny has the best pitch support (77%),
pYIN the best chroma/coverage; NNLS is clearly weakest (59% support). Onset grid lock is
~0.1 for every method *and* for raw onset detectors — sung syllables are genuinely loose,
so vocals need rhythmic quantisation / user correction rather than a better detector.

**Other**: no two methods agree (F1 ≤ 0.33), confirming it is mostly pads and ear candy.
Basic Pitch gives the most plausible notes (93% support, ~3.8 voices, 25x RT); the
piano model only fires on piano-like attacks (79% coverage, 0.9x RT, 43M params); NNLS
produces ~14 simultaneous notes. Treat this stem as harmony context (chroma/chords), not
as a source of notes. With EchoGrid's 6-stem split, re-run `piano_hr` on `piano.wav`.

**Key awareness.** The key is estimated with Krumhansl-Kessler profiles on the chroma of
all pitched stems (Bb major, confidence 0.98; cached in `outputs/<song>/key.json`).
30 s windows all read Bb major except 210–270 s (D minor, likely a vi/iii section
sharing the same scale, so it is not treated as a modulation). Before key cleaning,
out-of-key notes were mostly E, F#, Ab and B, and much shorter than in-key notes
(bass pYIN median 185 ms vs. 406 ms), which supports the "it's a bend/slide" hypothesis.
Rule: a non-diatonic semitone run held < 0.25 s that is not in the local harmony
('other' stem chroma ≥ 0.8) takes the pitch of the neighbouring in-key run within the
same phrase (ties → following note, i.e. the slide target). Key cleaning only relabels
pitches; note boundaries still come from the uncleaned contour, otherwise folding
created spurious legato splits.

| bass | notes | out-of-key time | pitch support | grid lock |
|---|---|---|---|---|
| pyin+onset | 515 | 3.6% | 86.8 | 0.306 |
| **pyin+onset+key** | 500 | 0.0% | 85.4 | **0.315** |
| crepe_tiny+onset | 477 | 1.5% | 88.9 | 0.324 |
| **crepe_tiny+onset+key** | 474 | 0.0% | 88.3 | **0.330** |
| nnls_baseline+onset | 559 | 4.9% | 86.4 | 0.292 |
| nnls_baseline+onset+key | 534 | 0.2% | 84.0 | 0.296 |

Vocals `+key`: out-of-key time 3–5.5% → ~0%, pitch support and chroma within ±1 point.
The small pitch-support drop is expected: during a glide the audio really *is* at the
passing pitch. The longest notes it relabelled (e.g. pYIN "Ab1" 430/615 ms at 66.8 s and
162.7 s) are G1→Eb2 bass glissandi in every model's contour, not sustained chromatic
notes. Caveats: genuine short chromatic passing tones (and bVII/secondary-dominant
roots shorter than 0.25 s) will be folded; long glides become one note at an in-key
pitch where a pitch bend would be more faithful.

## Direction

- Bass: f0 contour from pYIN (no model) or CREPE-tiny (fine-tunable on CPU) +
  onset-driven segmentation. The app's NNLS pitch is already competitive; adopting
  attack-based segmentation is the cheapest big win, and key-aware folding a cheap
  second one (fewer, more on-grid notes, no out-of-key blips).
- Key awareness should be a soft prior, not a hard snap: let the user set/confirm the
  key in the app, and consider emitting detected glides (monotonic f0 drift ≥ 2
  semitones within a note) as pitch bends instead of folding them.
- Vocals: CREPE-tiny contour + segmentation by pitch change, then grid quantisation.
- Other: Basic Pitch for candidate notes and a chord model for harmony.
- CPU fine-tuning candidates: Basic Pitch (36k), PESTO (130k), CREPE-tiny (0.5M).
  Train base models (e.g. a Basic-Pitch-sized net on Demucs-separated Slakh/MoisesDB
  stems, or MT3/YourMT3-class models) on the Spark; ship only small models here.
- Next: export user-corrected MIDI from the app as ground truth so this harness can
  report real precision/recall instead of proxies.

## Drums

Drums have no pitch, so this is automatic drum transcription (ADT): detect onsets and
classify each into one of 5 classes (kick, snare, hi-hat, tom, cymbal → GM notes 36/38/42/47/49).

### Literature → implementations tried

| key | paper | implementation | type |
|---|---|---|---|
| `adtof` | Zehren, Alunno & Bientinesi, *ADTOF: a large dataset of non-synthetic music for automatic drum transcription*, ISMIR 2021; *High-quality and reproducible ADT from crowdsourced data*, Signals 2023 | [xavriley/ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) (port of [MZehren/ADTOF](https://github.com/MZehren/ADTOF)) | Frame-level CRNN on log-spectrogram, 5 classes, peak-picking; ~450k params. **Weights CC BY-NC-SA (non-commercial).** |
| `yourmt3` | Chang, Benetos, Kirchhoff & Dixon, *YourMT3+: Multi-instrument music transcription with enhanced transformer architectures and cross-dataset stem augmentation*, IEEE MLSP 2024 (arXiv 2407.04822) | [mimbres/YourMT3](https://github.com/mimbres/YourMT3) via the [`mt3-infer`](https://pypi.org/project/mt3-infer/) package | Seq-to-seq Transformer (MT3 family, MoE + RoPE) emitting MIDI tokens for all instruments; drum notes kept. ~134M params, separate `.venv-mt3`. |
| `nmfd` | Dittmar & Müller, *Reverse engineering the Amen break — score-informed separation and restoration applied to drum recordings*, IEEE/ACM TASLP 2016 (NMFD after Smaragdis 2004) | [groupmm/libnmfd](https://github.com/groupmm/libnmfd) (MIT) | No training: non-negative matrix factor *deconvolution* with 5 spectro-temporal templates initialised from drum shapes, then per-class peak picking, kick/tom and hihat/cymbal exclusivity, and snapping to stem attacks. |

Background survey: Wu et al., *A review of automatic drum transcription*, IEEE/ACM TASLP 2018.
Not tried: Vogl et al. CRNN (no maintained code), Magenta Onsets-and-Frames drums
(TF1), Omnizart (TF, Python ≤ 3.8), DrummerScore (notebooks), and RunarFosse's ADT thesis
(training code, no released weights).

### Run

```powershell
.venv\Scripts\python.exe run_drums.py             # adtof, yourmt3, nmfd on the full drum stem
.venv\Scripts\python.exe evaluate_drums.py        # metrics, lane plots, A/B + synth audio
start "outputs\Firm Foundation (He Won't)\drums\drums_report.html"
```

`--methods`, `--start/--duration`, `--force` and `--song-dir/--bpm` work as for the pitch
runner. `run_config.json` is rewritten on every run, so do a full-song run last.
MIDI: `outputs/<song>/drums/runs/<method>.mid` (GM channel 10).

### Findings — Firm Foundation drum stem (382 s)

| method | hits | kick | snare | hihat | tom | cymbal | attack recall / precision | grid lock | speed (CPU) |
|---|---|---|---|---|---|---|---|---|---|
| adtof | 883 | 328 | 79 | 162 | 256 | 58 | 87 / **95** % | **0.75** | **31× realtime** |
| yourmt3 | 1749 | 369 | 162 | 672 | 301 | 160 | **96** / 51 % | 0.53 | 0.8× realtime |
| nmfd | 872 | 330 | 154 | 159 | 139 | 90 | 81 / 76 % | 0.53 | 1.0× realtime |

Attack recall and precision are measured against SuperFlux onsets in the stem at 30 ms.
Grid lock is for the 16th-triplet grid; the stem itself scores 0.72.
Pairwise onset F1 (50 ms) for adtof vs yourmt3: kick 0.94, tom 0.65, snare 0.48, cymbal 0.44, hihat 0.27.

- **Kick is solved**: all three agree, and ADTOF/YourMT3 match almost exactly.
- **ADTOF** gives the cleanest, most on-grid output and is ~30× faster than realtime.
  Its weakness here is **snare vs tom**: about half of YourMT3's snares are labelled tom
  by ADTOF, which explains its low snare count (79) and high tom count (256). Its
  hi-hat lane is sparse (it misses the quiet ride/hat pattern).
- **YourMT3** catches nearly every attack (including ghost notes and the hi-hat
  pattern) but over-generates (precision 51 %), with 19 different GM drum notes
  (tambourine, splash, etc.) and occasional pitched bleed. It's slow, but still usable on CPU.
- **NMFD** needs no training data and adapts to the kit, but cymbal/hi-hat cross-talk
  and threshold sensitivity make it the least reliable. It's useful as an unsupervised
  "user kit adaptation" idea rather than a main detector.
- Direction: ADTOF-style CRNN as the base detector (fast enough for CPU fine-tuning
  on user corrections; retrain on the Spark on Demucs-separated drum stems to fix the
  licence and the snare/tom confusion), optionally with YourMT3 as a slow "high recall"
  second opinion.
