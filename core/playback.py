"""
core/playback.py
================
Real-time MIDI + audio playback using pygame.
Non-blocking background thread.
"""

import threading
import time
import numpy as np
import pygame
import pygame.midi
import pygame.mixer


def _find_output_device() -> int:
    for i in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(i)
        if info[3]:
            return i
    return pygame.midi.get_default_output_id()


class MidiPlayback:
    """Non-blocking MIDI + audio playback."""

    VELOCITY = 90
    CHANNEL = 0
    PROGRAM = 0
    MIXER_FREQ = 44100

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=self.MIXER_FREQ, size=-16, channels=2, buffer=1024)
        pygame.midi.init()
        self._midi_port = None
        try:
            dev_id = _find_output_device()
            self._midi_port = pygame.midi.Output(dev_id, latency=0)
            self._midi_port.set_instrument(self.PROGRAM, self.CHANNEL)
        except Exception as exc:
            print(f"[MIDI] Could not open output: {exc}")

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, blocks: list, t_start: float,
             audio_slice: np.ndarray | None = None, sr: int = 44100) -> None:
        self.stop()
        self._stop_event.clear()
        sound = self._make_sound(audio_slice, sr) if audio_slice is not None else None
        self._thread = threading.Thread(
            target=self._run, args=(blocks, t_start, sound), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        self._thread = None
        if pygame.mixer.get_init():
            pygame.mixer.stop()
        if self._midi_port is not None:
            for note in range(128):
                try:
                    self._midi_port.note_off(note, 0, self.CHANNEL)
                except Exception:
                    pass

    def _make_sound(self, audio_slice: np.ndarray, sr: int):
        try:
            mixer_init = pygame.mixer.get_init()
            if mixer_init and mixer_init[0] != sr:
                pygame.mixer.quit()
                pygame.mixer.init(frequency=sr, size=-16, channels=2, buffer=1024)
            elif not mixer_init:
                pygame.mixer.init(frequency=sr, size=-16, channels=2, buffer=1024)
            peak = np.max(np.abs(audio_slice))
            if peak < 1e-6:
                return None
            audio_int16 = ((audio_slice / peak) * 32767).astype(np.int16)
            stereo = np.column_stack([audio_int16, audio_int16])
            return pygame.sndarray.make_sound(stereo)
        except Exception as exc:
            print(f"[Audio] Could not prepare sound: {exc}")
            return None

    def _run(self, blocks, t_start, sound):
        port = self._midi_port
        if port is None:
            if sound is not None:
                sound.play()
                while sound.get_num_channels() > 0 and not self._stop_event.is_set():
                    time.sleep(0.05)
            return
        try:
            events = []
            for entry in blocks:
                t0, t1, midi, *_ = entry
                if midi is None:
                    continue
                events.append((t0 - t_start, "on", midi))
                events.append((t1 - t_start, "off", midi))
            events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

            active_notes: set[int] = set()
            if sound is not None:
                sound.play()
            wall_start = time.perf_counter()

            for ev_time, ev_type, note in events:
                target = wall_start + ev_time
                while True:
                    now = time.perf_counter()
                    if now >= target:
                        break
                    remaining = target - now
                    if self._stop_event.is_set():
                        break
                    time.sleep(min(remaining * 0.5, 0.002))
                if self._stop_event.is_set():
                    break
                if ev_type == "on":
                    port.note_on(note, self.VELOCITY, self.CHANNEL)
                    active_notes.add(note)
                else:
                    port.note_off(note, 0, self.CHANNEL)
                    active_notes.discard(note)
        finally:
            for note in list(active_notes):
                try:
                    port.note_off(note, 0, self.CHANNEL)
                except Exception:
                    pass
            if sound is not None and self._stop_event.is_set():
                try:
                    sound.stop()
                except Exception:
                    pass
