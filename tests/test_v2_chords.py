"""Chord symbols, placeholder recognizer, timing session and lead-sheet export."""

from fractions import Fraction as F
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from audio_to_midi import chords, Grid
from audio_to_midi.beats import Beat
from audio_to_midi.domain import ChordSegment, Note, Project
from audio_to_midi.timing import BeatSession
from audio_to_midi.application import analyze_harmony, detect_key
from audio_to_midi.spectrogram import cqt_image, HOP_LENGTH, SAMPLE_RATE


class SymbolTests(unittest.TestCase):
    def test_parse_aliases_and_slash(self):
        self.assertEqual(chords.parse("Cmaj7"), (0, "maj7", None))
        self.assertEqual(chords.parse("Bbm7b5"), (10, "m7b5", None))
        self.assertEqual(chords.parse("F#-7"), (6, "m7", None))
        self.assertEqual(chords.parse("C/E"), (0, "", 4))
        self.assertEqual(chords.parse("C/C"), (0, "", None))
        self.assertIsNone(chords.parse("N.C."))
        self.assertEqual(chords.normalize("A#min"), "Bbm")
        with self.assertRaises(ValueError):
            chords.parse("H7")
        with self.assertRaises(ValueError):
            chords.parse("C13")

    def test_spell_follows_key_without_changing_stored_symbol(self):
        self.assertEqual(chords.normalize("G#m7"), "Abm7")
        self.assertEqual(chords.spell("Abm7", "B"), "G#m7")
        self.assertEqual(chords.spell("Bb7/Ab", "Eb"), "Bb7/Ab")
        self.assertEqual(chords.spell("Ebm/Bb", "B"), "D#m/A#")
        self.assertEqual(chords.spell("Fdim", "F#"), "E#dim")
        self.assertEqual(chords.spell("Ab", ""), "Ab")
        self.assertEqual(chords.spell("N.C.", "B"), "N.C.")
        self.assertEqual(chords.normalize(chords.spell("Ebm/Bb", "B")), "Ebm/Bb")

    def test_voicing_puts_slash_bass_low(self):
        self.assertEqual(chords.voicing("C/E"), [40, 60, 64, 67])
        self.assertEqual(chords.voicing("Am7"), [45, 69, 72, 76, 79])
        self.assertEqual(chords.voicing("N.C."), [])

    def test_name_notes(self):
        self.assertEqual(chords.name_notes([48, 64, 67, 72]), "C")
        self.assertEqual(chords.name_notes([52, 60, 67]), "C/E")
        self.assertEqual(chords.name_notes([57, 60, 64, 67]), "Am7")
        self.assertEqual(chords.name_notes([]), "N.C.")

    def test_segment_validates_and_normalizes(self):
        segment = ChordSegment(F(0), F(2), "Dbmaj7", alternatives=("C#",))
        self.assertEqual((segment.symbol, segment.alternatives), ("C#maj7", ("C#",)))
        with self.assertRaises(ValueError):
            ChordSegment(F(1), F(1), "C")
        with self.assertRaises(ValueError):
            Project("id", "x", Grid(), chords=[ChordSegment(F(0), F(2), "C"), ChordSegment(F(1), F(3), "G")])


class RecognizerTests(unittest.TestCase):
    def image(self, frames, low=21):
        return np.zeros((88, frames), dtype=np.float32), low

    def test_recognizes_triad_and_slash_bass_per_beat_then_merges(self):
        cqt, low = self.image(400)
        for pitch in (60, 64, 67):
            cqt[pitch - low, :200] = 1.0
        cqt[36 - low, :200] = 0.9  # C bass
        for pitch in (55, 59, 62):
            cqt[pitch - low, 200:] = 1.0
        cqt[47 - low, 200:] = 0.9  # B bass under G
        grid = Grid(120, 4, 4)
        segments = chords.recognize(cqt, low, 0.01, grid, 4.0)
        self.assertEqual([(s.start, s.end, s.symbol) for s in segments], [
            (F(0), F(4), "C"), (F(4), F(8), "G/B"),
        ])
        self.assertTrue(segments[0].alternatives)

    def test_bass_slide_into_the_root_is_not_a_slash_bass(self):
        # Measured on a bass stem: F#1 held after a slide from E1; F1 is nearly as strong.
        low = 21
        power = np.zeros((88, 10), dtype=np.float32)
        power[28 - low], power[29 - low], power[30 - low], power[42 - low] = 0.05, 0.99, 1.0, 0.54
        self.assertEqual(chords._bass(power, low), 6)
        power[29 - low] = 0.3  # a genuinely separate lower note still wins
        power[30 - low] = 0.0
        power[42 - low] = 1.0
        self.assertEqual(chords._bass(power, low), 5)

    def test_kick_drum_thump_is_not_a_slash_bass(self):
        # E major with a sustained E2 bass; a G1 kick rings for 15 of 50 frames each beat.
        cqt, low = self.image(400)
        for pitch in (40, 52, 59, 64, 68, 71):
            cqt[pitch - low] = 1.0
        for start in range(0, 400, 50):
            cqt[31 - low, start:start + 15] = 1.0
        segments = chords.recognize(cqt, low, 0.01, Grid(120, 4, 4), 4.0)
        self.assertEqual([s.symbol for s in segments], ["E"])

    def test_band_chroma_splits_registers_and_sums_to_what_key_detection_reads(self):
        cqt, low = self.image(50)
        for pitch, level in ((40, 1.0), (59, 0.9), (64, 0.9), (83, 0.8)):  # E2 | B3 E4 | B5
            cqt[pitch - low] = level
        profiles = chords.band_chroma(cqt, low)
        self.assertEqual(set(profiles), {"Low", "Mid", "High", "All"})
        np.testing.assert_allclose(profiles["Low"] + profiles["Mid"] + profiles["High"], profiles["All"])
        self.assertAlmostEqual(float(profiles["All"].sum()), 1.0)
        self.assertEqual(int(profiles["Low"].argmax()), 4)
        self.assertEqual(set(np.flatnonzero(profiles["Mid"])), {4, 11})
        self.assertEqual(set(np.flatnonzero(profiles["High"])), {11})
        folded = chords.fold_chroma(chords.harmonic_power(cqt), low).mean(axis=1)
        np.testing.assert_allclose(profiles["All"], folded / folded.sum(), rtol=1e-5)
        self.assertTrue(all(not v.any() for v in chords.band_chroma(np.zeros_like(cqt), low).values()))

    def test_silence_is_no_chord(self):
        cqt, low = self.image(100)
        segments = chords.recognize(cqt, low, 0.01, Grid(120, 4, 4), 1.0)
        self.assertEqual([s.symbol for s in segments], ["N.C."])

    def test_audio_distinguishes_major_minor_and_bass_even_with_octave_doublings(self):
        time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        for third, expected in ((70, "F#"), (69, "F#m")):
            with self.subTest(chord=expected):
                audio = sum(
                    0.2 * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * time)
                    for pitch in (42, 54, 61, 66, third, 73)
                ).astype(np.float32)
                image = cqt_image(audio, SAMPLE_RATE)
                segments = chords.recognize(
                    image, 21, HOP_LENGTH / SAMPLE_RATE, Grid(), 2.0, key="B",
                )
                self.assertEqual([s.symbol for s in segments], [expected])

    def test_key_resolves_missing_third_without_forcing_borrowed_minor(self):
        image, low = self.image(100)
        for pitch in (42, 61, 66):
            image[pitch - low] = 1.0
        major = chords.recognize(image, low, 0.01, Grid(), 1.0, key="B major")
        minor = chords.recognize(image, low, 0.01, Grid(), 1.0, key="Bm")
        self.assertEqual([s.symbol for s in major], ["F#"])
        self.assertEqual([s.symbol for s in minor], ["F#m"])
        image[69 - low] = 1.0
        borrowed = chords.recognize(image, low, 0.01, Grid(), 1.0, key="B")
        self.assertEqual([s.symbol for s in borrowed], ["F#m"])

    def test_weak_third_is_not_discarded(self):
        image, low = self.image(100)
        for pitch, level in ((42, 1.0), (61, 0.95), (69, 0.85)):
            image[pitch - low] = level
        segments = chords.recognize(image, low, 0.01, Grid(), 1.0, key="B")
        self.assertEqual([s.symbol for s in segments], ["F#m"])

    def test_key_preserves_supported_extended_and_suspended_chords(self):
        for expected in ("F#7", "F#m7b5", "F#dim", "F#sus4", "Bmaj7"):
            with self.subTest(chord=expected):
                image, low = self.image(100)
                root, quality, _ = chords.parse(expected)
                image[36 + root - low] = 1.0
                for interval in chords.QUALITIES[quality]:
                    image[60 + root + interval - low] = 1.0
                segments = chords.recognize(image, low, 0.01, Grid(), 1.0, key="B")
                self.assertEqual([s.symbol for s in segments], [expected])

    def test_harmony_uses_explicit_or_detected_key(self):
        image, low = self.image(100)
        for pitch in (42, 61, 66):
            image[pitch - low] = 1.0
        for supplied, expected_key, expected_chord in (("Bm", "Bm", "F#m"), ("", "B", "F#")):
            with self.subTest(key=supplied), patch(
                "audio_to_midi.application.mix_cqt", return_value=(image, 0.01, 1.0),
            ), patch("audio_to_midi.application.detect_key", return_value="B"):
                project = Project("id", "Song", Grid(), mix_path="mix.wav", key=supplied)
                key, segments = analyze_harmony(project)
                self.assertEqual(key, expected_key)
                self.assertEqual([s.symbol for s in segments], [expected_chord])

    def test_detect_key_from_b_major_progression_and_reject_silence(self):
        image, low = self.image(800)
        for i, pitches in enumerate(((47, 63, 66), (40, 56, 59), (42, 58, 61), (44, 59, 63))):
            for pitch in pitches:
                image[pitch - low, i * 200:(i + 1) * 200] = 1.0
        self.assertEqual(detect_key(image), "B")
        with self.assertRaisesRegex(ValueError, "No pitched harmony"):
            detect_key(np.zeros_like(image))

    def test_notes_to_segments_names_block_chords(self):
        notes = [Note(f"n{p}", F(0), F(4), p) for p in (48, 52, 55)] + [
            Note(f"m{p}", F(4), F(8), p) for p in (45, 52, 57, 60)
        ]
        segments = chords.notes_to_segments(notes, Grid())
        self.assertEqual([(s.start, s.end, s.symbol) for s in segments], [(F(0), F(4), "C"), (F(4), F(8), "Am")])

    def test_musicxml_is_well_formed_with_harmony_per_change(self):
        segments = [ChordSegment(F(0), F(6), "C/E"), ChordSegment(F(6), F(8), "Bb7")]
        root = ET.fromstring(chords.musicxml(segments, Grid(100, 4, 4), "Song").split("\n", 2)[2])
        harmonies = root.findall(".//harmony")
        self.assertEqual(len(harmonies), 3)  # C/E, C/E restated at bar 2, Bb7
        self.assertEqual(harmonies[2].find("root/root-alter").text, "-1")
        self.assertEqual(harmonies[0].find("bass/bass-step").text, "E")
        self.assertEqual(len(root.findall(".//measure")), 2)

    def test_musicxml_spells_chords_and_signature_in_key(self):
        segments = [ChordSegment(F(0), F(4), "Abm"), ChordSegment(F(4), F(8), "Gb/Bb")]
        root = ET.fromstring(chords.musicxml(segments, Grid(100, 4, 4), "Song", key="B").split("\n", 2)[2])
        self.assertEqual(root.find(".//key/fifths").text, "5")
        harmonies = root.findall(".//harmony")
        self.assertEqual(
            [(h.find("root/root-step").text, h.find("root/root-alter").text) for h in harmonies],
            [("G", "1"), ("F", "1")],
        )
        self.assertEqual(harmonies[1].find("bass/bass-step").text, "A")
        self.assertEqual(harmonies[1].find("bass/bass-alter").text, "1")
        self.assertEqual(harmonies[0].find("kind").get("text"), "m")

    def test_detected_key_uses_the_common_enharmonic(self):
        image, low = np.zeros((88, 800), dtype=np.float32), 21
        # Ab minor progression: Abm, Dbm, Eb, Abm (detector names the tonic Ab).
        for i, pitches in enumerate(((44, 59, 63), (49, 64, 68), (51, 55, 58), (44, 59, 63))):
            for pitch in pitches:
                image[pitch - low, i * 200:(i + 1) * 200] = 1.0
        self.assertEqual(detect_key(image), "G#m")


class StemHarmonyTests(unittest.TestCase):
    """Chords from separated stems: no drums, bass from the bass stem, vocals only for the key."""

    def setUp(self):
        import tempfile

        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE

    def tone(self, *pitches, level=0.2):
        audio = np.zeros_like(self.time)
        for pitch in pitches:
            audio += level * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * self.time)
        return audio.astype(np.float32)

    def project(self, stems, key="E"):
        import soundfile as sf
        from pathlib import Path
        from audio_to_midi.application import add_stem

        root = Path(self.folder.name)
        mix = sum(stems.values())
        sf.write(root / "mix.wav", mix, SAMPLE_RATE)
        project = Project("id", "Song", Grid(), mix_path=str(root / "mix.wav"), key=key)
        for role, audio in stems.items():
            sf.write(root / f"{role}.wav", audio, SAMPLE_RATE)
            add_stem(project, root / f"{role}.wav", role)
        return project

    def symbols(self, project):
        return [s.symbol for s in analyze_harmony(project)[1]]

    def test_drums_and_vocal_melody_do_not_shape_stem_chords(self):
        project = self.project({
            "piano": self.tone(64, 68, 71),
            "bass": self.tone(40, level=0.3),
            "drums": self.tone(31, level=0.6),  # pitched low thump at G1
            "vocals": self.tone(73, 74, level=0.5),  # loud C#5/D5 melody outside E major triad
        })
        self.assertEqual(self.symbols(project), ["E"])
        project.tracks = []
        self.assertNotEqual(self.symbols(project), ["E"])  # the mix alone is misled

    def test_slash_bass_comes_from_bass_stem(self):
        project = self.project({"guitar": self.tone(64, 68, 71), "bass": self.tone(44, level=0.3)})
        self.assertEqual(self.symbols(project), ["E/Ab"])  # stored canonically; shown as E/G# in E

    def test_vocals_carry_harmony_when_instrument_stems_are_silent(self):
        silent = np.zeros_like(self.time, dtype=np.float32)
        project = self.project({"piano": silent, "vocals": self.tone(64, 68, 71)})
        self.assertEqual(self.symbols(project), ["E"])

    def test_loud_bass_on_the_iv_root_does_not_pull_the_key(self):
        from audio_to_midi.application import harmony_cqt

        # B major harmony (A# rules out E major) over a loud bass dwelling on E, the IV.
        stems = {"piano": self.tone(59, 63, 66, 70, 71), "bass": self.tone(28, 40, level=0.8)}
        project = self.project(stems, key="")
        self.assertNotEqual(detect_key(harmony_cqt(project)[0]), "B")  # chord audio, bass included
        self.assertEqual(analyze_harmony(project)[0], "B")

    def test_stale_or_missing_stems_fall_back_to_the_mix(self):
        from audio_to_midi.application import harmony_sources, stems_for_chords

        project = self.project({"piano": self.tone(64, 68, 71), "bass": self.tone(40, level=0.3)})
        self.assertTrue(stems_for_chords(project))
        self.assertTrue(harmony_sources(project).from_stems)
        project.stages["preparation"] = {"warp_revision": "newer"}
        self.assertFalse(stems_for_chords(project))
        self.assertEqual(harmony_sources(project).chords, (project.mix_path,))


class BtcTests(unittest.TestCase):
    """BTC chord transformer: label mapping, beat folding and (when downloaded) the real weights."""

    def test_labels_fold_btc_vocabulary_onto_app_symbols(self):
        from audio_to_midi import btc

        names = btc.labels()
        self.assertEqual(len(names), 170)
        self.assertEqual(names[0], "Cm")
        self.assertEqual(names[1], "C")
        self.assertEqual(names[14 * 6 + 4], "F#m")  # F# min6 -> F#m
        self.assertEqual(names[14 * 11 + 11], "Bm7b5")  # B hdim7
        self.assertEqual(names[14 * 9 + 7], "AmMaj7")
        self.assertIsNone(names[168])
        self.assertEqual(names[169], chords.NO_CHORD)
        for name in names:
            if name is not None:
                chords.parse(name)

    def test_beats_take_the_most_probable_symbol_and_bass_inversions(self):
        from audio_to_midi import btc

        grid = Grid()  # 120 BPM: one beat = 0.5 s
        times = np.arange(0, 2.0, 0.1)
        probs = np.zeros((len(times), 170), dtype=np.float32)
        e, f_sharp, e_min6 = 14 * 4 + 1, 14 * 6 + 1, 14 * 4 + 4
        probs[:10, e] = .5
        probs[:10, e_min6] = .3  # folds into Em, still below E
        probs[:10, 168] = .2  # 'X' never wins
        probs[10:, f_sharp] = 1
        bass = np.zeros((88, 200), dtype=np.float32)
        bass[44 - 21, 100:150] = 1.0  # G#2 on beat 3: not an F# chord tone, so no slash
        bass[46 - 21, 150:] = 1.0  # A#2 on beat 4: the third, so F#/A#
        with patch.object(btc, "probabilities", return_value=(probs, times)):
            segments = btc.recognize(np.zeros(1), grid, 2.0, bass, 21, 0.01)
        self.assertEqual([(s.start, s.end, s.symbol) for s in segments], [
            (F(0), F(2), "E"), (F(2), F(3), "F#"), (F(3), F(4), "F#/Bb"),
        ])
        self.assertAlmostEqual(segments[0].confidence, .5 / .8, places=5)
        self.assertIn("Em", segments[0].alternatives)

    def test_real_weights_recognize_synthesized_triads(self):
        from audio_to_midi import btc

        if not btc.checkpoint_ready():
            self.skipTest("BTC weights not downloaded")
        time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE

        def chord(*pitches):
            return sum(
                np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * k * time) / k ** 1.5
                for p in pitches for k in range(1, 7)
            ) * np.exp(-time * .3)

        audio = np.concatenate([chord(48, 60, 64, 67), chord(45, 57, 60, 64), chord(43, 55, 59, 62)])
        audio = (audio / np.abs(audio).max() / 2).astype(np.float32)
        segments = btc.recognize(audio, Grid(), len(audio) / SAMPLE_RATE)
        self.assertEqual([s.symbol for s in segments], ["C", "Am", "G"])
        self.assertEqual([s.start for s in segments], [F(0), F(4), F(8)])

    def test_missing_weights_are_reported_for_download(self):
        from pathlib import Path
        from audio_to_midi import btc_paths, provision

        with patch.object(btc_paths, "CHECKPOINT", Path("does-not-exist.pt")):
            self.assertIn("BTC chord weights (~12 MB)", provision.missing("btc"))

    def test_package_install_retries_once_on_the_configured_feed(self):
        from audio_to_midi import provision
        from audio_to_midi.pipeline import PreparationError

        calls = []

        def flaky(argv, **kwargs):
            calls.append(argv)
            if len(calls) == 1:
                raise PreparationError("Installing x failed (exit 1):\nERROR: HTTP error 404")

        installed = iter([False, True])
        with patch.dict(provision.FEATURES, {"x": [("x_mod", ["x>=1"])]}), \
                patch.object(provision, "_run_process", side_effect=flaky), \
                patch.object(provision, "_installed", side_effect=lambda module: next(installed)):
            provision.ensure("x")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertFalse(any("--index-url" in arg or "--extra-index-url" in arg for arg in calls[0]))

        def broken(argv, **kwargs):
            raise PreparationError("Installing x failed (exit 1)")

        with patch.dict(provision.FEATURES, {"x": [("x_mod", ["x>=1"])]}), \
                patch.object(provision, "_run_process", side_effect=broken) as process, \
                patch.object(provision, "_installed", return_value=False):
            with self.assertRaises(PreparationError):
                provision.ensure("x")
        self.assertEqual(process.call_count, 2)

    def test_analyze_harmony_routes_to_btc_with_stem_audio(self):
        from audio_to_midi import btc

        stems = StemHarmonyTests("test_slash_bass_comes_from_bass_stem")
        stems.setUp()
        self.addCleanup(stems.doCleanups)
        project = stems.project({"guitar": stems.tone(64, 68, 71), "bass": stems.tone(40, level=0.3)})
        seen = {}

        def fake(audio, grid, duration, bass_cqt, low, frame_seconds, cancel):
            seen.update(samples=len(audio), bass=bass_cqt is not None, duration=duration)
            return [ChordSegment(F(0), F(4), "E", "detected", 1.0)]

        with patch.object(btc, "recognize", fake):
            key, segments = analyze_harmony(project, model="btc")
        self.assertEqual((key, [s.symbol for s in segments]), ("E", ["E"]))
        self.assertEqual(seen, {"samples": SAMPLE_RATE * 2, "bass": True, "duration": 2.0})
        with self.assertRaises(ValueError):
            analyze_harmony(project, model="nope")


class BeatSessionTests(unittest.TestCase):
    def session(self):
        beats = [Beat(0.5 * i, 0.9) for i in range(48)]
        return BeatSession(beats, 120.0, 24.0)

    def test_two_downbeat_pins_count_bars_and_make_grid(self):
        session = self.session()
        session.pin(2, 1)
        session.pin(10, 3)
        self.assertAlmostEqual(session.fit.bpm, 120, places=3)
        self.assertEqual(session.mark_of(6), (2, 1))
        grid = session.grid(F(1, 4))
        self.assertAlmostEqual(grid.offset_seconds, 1.0)
        self.assertEqual(session.target_bpm(), 120)
        self.assertGreater(len(session.warp_anchors("smooth")), 2)
        with self.assertRaises(ValueError):
            session.pin(4, 1)  # bar 1 is already pinned elsewhere

    def test_meter_change_keeps_marks_and_rejects_impossible_beats(self):
        session = self.session()
        session.pin(2, 1)
        session.pin(5, 1, 4)
        with self.assertRaises(ValueError):
            session.set_meter(3, 4)
        self.assertEqual((session.numerator, session.denominator), (4, 4))
        restored = BeatSession.from_json(session.to_json(), 4, 4)
        self.assertEqual([b.mark for b in restored.pins], [(1, 1), (1, 4)])

    def test_taps_on_downbeats_pin_bars(self):
        session = self.session()
        taps = [1.0 + 2.0 * bar + 0.04 for bar in range(8)]
        result = session.apply_taps(taps, session.tap_first_bar(taps[0]))
        self.assertEqual(result.meter, (4, 4))
        self.assertEqual(session.mark_of(2), (1, 1))
        self.assertAlmostEqual(session.fit.downbeat, 1.0)
        self.assertIn("Taps", session.summary())


if __name__ == "__main__":
    unittest.main()
