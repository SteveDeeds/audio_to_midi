import os
import tempfile
import unittest
from types import SimpleNamespace

from core.session import SongSession


class TestSongSession(unittest.TestCase):
    def setUp(self) -> None:
        handle, self.path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def markers(self, tag: str = "original") -> SimpleNamespace:
        return SimpleNamespace(wav_path=self.path, tag=tag, edits=[])

    def test_cached_stem_reuses_analysis_and_edits(self) -> None:
        song = SongSession()
        markers = self.markers()
        song.store(markers, 120.0, 4)
        markers.edits.append("C4")

        restored = song.activate_cached(self.path, 120.0, 4)

        self.assertIsNotNone(restored)
        self.assertIs(restored.markers, markers)
        self.assertEqual(restored.markers.edits, ["C4"])

    def test_grid_change_requires_new_analysis(self) -> None:
        song = SongSession()
        song.store(self.markers(), 120.0, 4)

        self.assertIsNone(song.activate_cached(self.path, 121.0, 4))

    def test_file_change_invalidates_cached_analysis(self) -> None:
        song = SongSession()
        song.store(self.markers(), 120.0, 4)
        with open(self.path, "ab") as stream:
            stream.write(b"changed")

        self.assertIsNone(song.activate_cached(self.path, 120.0, 4))

    def test_lru_evicts_inactive_stem(self) -> None:
        song = SongSession(max_cached_stems=1)
        song.store(self.markers("first"), 120.0, 4)
        second_handle, second_path = tempfile.mkstemp(suffix=".wav")
        os.close(second_handle)
        self.addCleanup(
            lambda: os.path.exists(second_path) and os.remove(second_path)
        )
        song.store(SimpleNamespace(wav_path=second_path), 120.0, 4)

        self.assertIsNone(song.activate_cached(self.path, 120.0, 4))
        self.assertEqual(song.active.markers.wav_path, second_path)


if __name__ == "__main__":
    unittest.main()