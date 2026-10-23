"""Project adapters for the shared transcription providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi import velocity
from core.models import CorrectionModel
from core import onset_features


MODELS = {
    "bass": ("crepe_tiny", "crepe_full"),
    "vocals": ("pyin", "crepe_full", "crepe_tiny"),
    "piano": ("basic_pitch", "piano_hr"),
    "guitar": ("basic_pitch", "piano_hr"),
    "drums": ("adtof",),
    "other": (),
}


def normalize_key(value: str) -> str:
    if not value.strip():
        return ""
    parts = value.strip().split()
    token = parts[0]
    minor = token.endswith("m")
    tonic = token[:-1] if minor else token
    if tonic not in {"C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"}:
        raise ValueError("Key must be a note name with optional major/minor, e.g. Bb major or Am.")
    if len(parts) > 1:
        if len(parts) == 2 and parts[1].lower() in ("major", "minor"):
            minor = parts[1].lower() == "minor"
        elif not (parts[1].startswith("(") and parts[-1].endswith(")")):
            raise ValueError("Unrecognized key mode.")
    return tonic + ("m" if minor else "")


class Cancelled(Exception):
    """A request was cancelled without publishing a result."""


@dataclass(frozen=True)
class Prediction:
    start: Fraction
    end: Fraction
    notes: tuple[Note, ...]
    onsets: tuple[Fraction, ...]


class TrackBackend:
    """One worker-owned audio/provider cache per track and settings identity."""

    def __init__(self, track: Track, project: Project):
        self.audio, self.sr = sf.read(track.audio_path, dtype="float32", always_2d=True)
        if not np.all(np.isfinite(self.audio)):
            raise ValueError("Stem contains non-finite audio samples.")
        self.audio = self.audio.mean(axis=1)
        self.audio = librosa.resample(self.audio, orig_sr=self.sr, target_sr=44100)
        self.sr = 44100
        peak = float(np.max(np.abs(self.audio)))
        if peak > 1e-6:
            self.audio /= peak
        self.grid = project.grid
        self.role = track.role
        self.model = track.model
        self.track_id = track.id
        self.duration = len(self.audio) / self.sr
        self.project_key = project.key
        self.provider = None
        self._feature_data = None
        self._cqt_data = None
        self._drum_levels = None
        self._path = track.audio_path
        self._license_acknowledged = track.settings.get("license_acknowledged", False)
        self.features: dict[Fraction, tuple[float, float, float, float]] = {}
        self.attacks: np.ndarray | None = None
        self._harmony_paths = (
            [project.mix_path] if project.mix_path else [
                t.audio_path for t in project.tracks if t.role != "drums"
            ]
        )
        self._other_path = next((t.audio_path for t in project.tracks if t.role == "other"), None)

    def _check_provider(self) -> None:
        if self.role == "other":
            raise ValueError("Other is harmony context only; it has no MIDI provider.")
        if self.model not in MODELS[self.role]:
            raise ValueError(f"Unsupported {self.role} model: {self.model}")
        if self.provider is not None:
            return
        if (self.role == "drums" or self.model == "piano_hr") and not self._license_acknowledged:
            raise ValueError(
                "Acknowledge this backend's checkpoint license and possible first-use download "
                "before transcription (ADTOF is non-commercial; review piano-HR terms)."
            )
        if self.role in ("bass", "vocals"):
            from core.mono_pitch import MonoTranscriber
            from core.chord_analysis import key_to_scale_pcs

            provider = MonoTranscriber(self.audio, self.sr, self.track_id)
            # Supply explicit project harmony rather than filename-based siblings.
            key = normalize_key(self.project_key or self._detect_project_key())
            scale = key_to_scale_pcs(key)
            if key.endswith("m"):
                tonic = librosa.note_to_midi(key.split()[0].rstrip("m") + "4") % 12
                scale.add((tonic + 11) % 12)
            harmony_times, harmony = np.array([]), np.zeros((12, 0))
            if self._other_path:
                from scipy import ndimage

                other, _ = librosa.load(self._other_path, sr=22050)
                harmony = librosa.feature.chroma_cqt(y=other, sr=22050, hop_length=2048, norm=None)
                harmony = ndimage.uniform_filter1d(harmony, size=max(1, int(.8 * 22050 / 2048)), axis=1)
                level = harmony.sum(axis=0)
                harmony /= harmony.max(axis=0, keepdims=True) + 1e-9
                harmony[:, level < .05 * np.percentile(level, 95)] = 0
                harmony_times = librosa.times_like(harmony, sr=22050, hop_length=2048)
            provider._context = scale, harmony_times, harmony
            provider._context_window = None
            self.provider = provider
        elif self.role in ("piano", "guitar"):
            from core.poly_pitch import PolyTranscriber

            self.provider = PolyTranscriber(self.audio, self.sr)
        else:
            from core.adtof_drums import AdtofDrumTranscriber

            self.provider = AdtofDrumTranscriber(self._path)

    @classmethod
    def create(cls, track: Track, project: Project) -> TrackBackend:
        return cls(track, project)

    def _detect_project_key(self) -> str:
        from core.analysis import detect_key_from_chroma

        if self.role == "bass" and self._harmony_paths == [self._path]:
            raise ValueError("Bass alone is unreliable for key detection. Set a key or import the full mix/other pitched stems.")
        parts = []
        for path in self._harmony_paths:
            audio, _ = librosa.load(path, sr=22050, duration=60)
            if np.any(audio):
                parts.append(librosa.feature.chroma_stft(y=audio, sr=22050).mean(axis=1))
        if not parts:
            raise ValueError("No pitched harmony is audible. Set the project key explicitly.")
        return detect_key_from_chroma(np.mean(parts, axis=0))

    def _candidate_features(self, beat: Fraction) -> tuple[float, float, float, float]:
        cached = self.features.get(beat)
        if cached is not None:
            return cached
        time = self.grid.beat_to_seconds(beat)
        half = float(self.grid.subdivision) * 30 / self.grid.bpm
        if self._feature_data is None:
            smooth = max(16, int(0.028 * self.sr))
            envelope = np.convolve(np.abs(self.audio).astype(float), np.ones(smooth) / smooth, mode="same")
            samples = np.arange(0, len(envelope), 512)
            chroma = librosa.feature.chroma_cqt(
                y=self.audio, sr=self.sr, hop_length=512, bins_per_octave=36,
            )
            cqt_times, magnitude = self._cqt()
            times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=self.sr, hop_length=512)
            self._feature_data = samples / self.sr, envelope[samples], times, chroma, cqt_times, magnitude
        env_times, envelope, times, chroma, cqt_times, magnitude = self._feature_data
        a, b = max(0.0, time - half), min(self.duration, time + half)
        transient = onset_features.transient_intensity(env_times, envelope, a, b)
        delta = onset_features.chroma_delta(times, chroma, time, half * 2)
        energy_db = onset_features.rms_db(self.audio, self.sr, a, b)
        spike = onset_features.percussive_spike(cqt_times, magnitude, time, half * 2)
        result = transient, delta, energy_db, spike
        self.features[beat] = result
        return result

    CQT_LOW_MIDI = 12

    def _cqt(self) -> tuple[np.ndarray, np.ndarray]:
        """Whole-stem semitone CQT magnitude (MIDI 12..94), computed once per backend."""
        if self._cqt_data is None:
            magnitude = np.abs(librosa.cqt(
                self.audio, sr=self.sr, hop_length=512,
                fmin=librosa.midi_to_hz(self.CQT_LOW_MIDI), n_bins=83,
            )).astype(np.float32)
            times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=self.sr, hop_length=512)
            self._cqt_data = times, magnitude, velocity.reference_power(magnitude)
        times, magnitude, _ = self._cqt_data
        return times, magnitude

    def _velocity(self, pitch: int, t0: float, t1: float) -> int:
        """Velocity from the CQT power of the note's first harmonics at its attack."""
        times, magnitude = self._cqt()
        power = velocity.harmonic_power(times, magnitude, self.CQT_LOW_MIDI, pitch, t0, t1)
        return velocity.velocity(power, self._cqt_data[2])

    def predict(
        self, track: Track, start: Fraction, end: Fraction,
        cancelled: Callable[[], bool],
    ) -> Prediction:
        if cancelled():
            raise Cancelled()
        self._check_provider()
        provider = self.provider
        grid = self.grid
        t0 = max(0.0, grid.beat_to_seconds(start))
        t1 = min(self.duration, grid.beat_to_seconds(end))
        if t1 <= t0:
            raise ValueError("Requested interval has no audio.")
        audio_start = min(end, max(start, grid.snap(grid.seconds_to_beat(t0))))
        if self.role == "drums":
            provider.prepare()
            # Replay only the current snapshot so undo also undoes adaptation.
            from core.adtof_drums import AdtofDrumTranscriber
            fresh = AdtofDrumTranscriber(self._path)
            fresh.set_activations(provider._act)
            self.provider = provider = fresh
            for label in track.training:
                if label.get("domain") == "drum":
                    provider.record_feedback(
                        float(label["start"]), float(label["end"]),
                        int(label["pitch"]), bool(label["keep"]),
                    )
            from core.adtof_drums import drum_class

            if self._drum_levels is None:
                self._drum_levels = velocity.DrumLevels(self.audio, self.sr)
            hits = list(provider.hits())
            levels = self._drum_levels.velocities([(hit.time, drum_class(hit.midi)) for hit in hits])
            notes = []
            for hit, level in zip(hits, levels):
                beat = max(Fraction(0), grid.snap(grid.seconds_to_beat(hit.time)))
                if start <= beat < end:
                    notes.append(self._note(beat, min(end, beat + grid.subdivision), hit.midi, level))
            if cancelled():
                raise Cancelled()
            return Prediction(start, end, tuple(notes), tuple(sorted({n.start for n in notes})))

        if not provider.prepare(t0, t1, self.role, self.model, should_cancel=cancelled):
            raise Cancelled()
        if cancelled():
            raise Cancelled()
        if self.attacks is None or self.role in ("piano", "guitar"):
            self.attacks = provider.onset_times(self.role, self.model)
        baseline = {
            max(Fraction(0), grid.snap(grid.seconds_to_beat(float(t))))
            for t in self.attacks
        }
        model = CorrectionModel()
        profile = track.settings.get("onset_profile")
        if profile:
            model.load_dict(profile, merge=False)
        labels = [row for row in track.training if row.get("domain") == "onset"]
        examples = []
        for label in labels:
            beat = Fraction(str(label["beat"]))
            examples.append((
                *self._candidate_features(beat), bool(label["keep"]),
                int(beat * 1000000), "user",
            ))
        if examples:
            model.record_candidate_batch(examples)
        active = set()
        cursor = audio_start
        while cursor < end:
            probability = model.candidate_keep_prob(*self._candidate_features(cursor)) if examples or profile else None
            if (probability is not None and probability >= 0.5) or (
                probability is None and cursor in baseline
            ):
                active.add(cursor)
            cursor += grid.subdivision
        for label in labels:
            beat = Fraction(str(label["beat"]))
            if bool(label["keep"]):
                active.add(beat)
            else:
                active.discard(beat)
        active.update(n.start for n in track.notes.authored if start <= n.start < end)
        # Held changes are an explicit backend policy, not raw f0 boundaries.
        if self.role in ("bass", "vocals") and track.settings.get("stable_pitch_changes", True):
            active.update(
                grid.snap(grid.seconds_to_beat(t))
                for t in provider.pitch_changes(t0, t1, self.role, self.model)
            )
        for label in labels:
            if not label["keep"]:
                active.discard(Fraction(str(label["beat"])))
        bounds = sorted({audio_start, end} | {b for b in active if audio_start <= b < end})
        notes: list[Note] = []
        for a, b in zip(bounds, bounds[1:]):
            if cancelled():
                raise Cancelled()
            ta, tb = max(t0, grid.beat_to_seconds(a)), min(t1, grid.beat_to_seconds(b))
            if self.role in ("bass", "vocals"):
                answer = provider.block_pitch(ta, tb, self.role, self.model)
                if answer is None:
                    continue
                pitches, release = answer
                stop = min(b, max(a + grid.subdivision, grid.snap(grid.seconds_to_beat(release))))
                candidate = self._note(a, stop, pitches[0], self._velocity(pitches[0], ta, release))
                if a == start and a not in active:
                    previous = next((
                        n for n in track.notes.generated
                        if n.pitch == candidate.pitch and n.end == a
                    ), None)
                    if previous:
                        candidate = Note(previous.id, a, stop, candidate.pitch, previous.velocity, "generated")
                notes.append(candidate)
            else:
                for answer in provider.block_notes(
                    ta, tb, self.role, self.model,
                    float(grid.subdivision) * 30 / grid.bpm + 1e-6,
                )[:int(track.settings.get("polyphony", 8))]:
                    stop = min(b, max(a + grid.subdivision, grid.snap(grid.seconds_to_beat(answer.release))))
                    previous = next((
                        n for n in reversed(notes) if n.pitch == answer.midi and n.end == a
                    ), None)
                    if answer.continues and previous:
                        notes.remove(previous)
                        notes.append(Note(
                            previous.id, previous.start, stop, previous.pitch,
                            previous.velocity, previous.source,
                        ))
                    else:
                        # piano_hr predicts real key velocities; Basic Pitch's amplitude is closer to confidence.
                        level = (
                            max(1, round(answer.velocity * 127)) if self.model == "piano_hr"
                            else self._velocity(answer.midi, ta, answer.release)
                        )
                        candidate = self._note(a, stop, answer.midi, level)
                        if answer.continues and a == start:
                            previous = next((
                                n for n in track.notes.generated
                                if n.pitch == answer.midi and n.end == a
                            ), None)
                            if previous:
                                candidate = Note(previous.id, a, stop, candidate.pitch, previous.velocity, "generated")
                        notes.append(candidate)
        return Prediction(start, end, tuple(notes), tuple(b for b in sorted(active) if start <= b < end))

    def _note(self, start: Fraction, end: Fraction, pitch: int, velocity: int) -> Note:
        return Note(
            f"{self.track_id}:{start}:{pitch}", start, end, pitch, velocity, "generated",
        )
