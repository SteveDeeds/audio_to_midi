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


MIDI_PAN_75_LEFT = 16
AUDIO_LEFT_GAIN = 0.25
AUDIO_RIGHT_GAIN = 1.0


def _midi_mix_messages(channel: int, gain: float) -> tuple[tuple[int, int, int], ...]:
    status = 0xB0 | int(channel)
    volume = int(round(127.0 * max(0.0, min(1.0, float(gain)))))
    return ((status, 10, MIDI_PAN_75_LEFT), (status, 7, volume))


def _stereo_audio_int16(audio_slice: np.ndarray, gain: float) -> np.ndarray | None:
    peak = float(np.max(np.abs(audio_slice)))
    if peak < 1e-6:
        return None
    normalized = audio_slice / peak * 32767.0 * max(0.0, min(1.0, float(gain)))
    return np.column_stack([
        normalized * AUDIO_LEFT_GAIN,
        normalized * AUDIO_RIGHT_GAIN,
    ]).astype(np.int16)


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
        self._stop_event: threading.Event | None = None
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
             audio_slice: np.ndarray | None = None, sr: int = 44100,
             channel: int | None = None, midi_gain: float = 1.0,
             audio_gain: float = 1.0) -> None:
        previous_thread = self._thread
        self.stop()
        if channel is None:
            channel = self.CHANNEL
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=self._run,
            args=(
                blocks, t_start, audio_slice, sr, channel, stop_event,
                previous_thread, float(midi_gain), float(audio_gain),
            ),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self._thread = None
        self._stop_event = None
        if pygame.mixer.get_init():
            pygame.mixer.stop()

    def _make_sound(self, audio_slice: np.ndarray, sr: int, gain: float):
        try:
            mixer_init = pygame.mixer.get_init()
            if mixer_init and mixer_init[0] != sr:
                pygame.mixer.quit()
                pygame.mixer.init(frequency=sr, size=-16, channels=2, buffer=1024)
            elif not mixer_init:
                pygame.mixer.init(frequency=sr, size=-16, channels=2, buffer=1024)
            stereo = _stereo_audio_int16(audio_slice, gain)
            if stereo is None:
                return None
            return pygame.sndarray.make_sound(stereo)
        except Exception as exc:
            print(f"[Audio] Could not prepare sound: {exc}")
            return None

    def _run(self, blocks, t_start, audio_slice, sr, channel, stop_event,
             previous_thread, midi_gain, audio_gain):
        sound = (
            self._make_sound(audio_slice, sr, audio_gain)
            if audio_slice is not None else None
        )
        if previous_thread is not None and previous_thread.is_alive():
            previous_thread.join()
        if stop_event.is_set():
            return
        if channel is None:
            channel = self.CHANNEL
        port = self._midi_port
        if port is None:
            if sound is not None:
                sound.play()
                while sound.get_num_channels() > 0 and not stop_event.is_set():
                    time.sleep(0.05)
            return
        try:
            for status, control, value in _midi_mix_messages(channel, midi_gain):
                port.write_short(status, control, value)
            events = []
            for entry in blocks:
                t0, t1, midi, *_ = entry
                if midi is None:
                    continue
                vel = int(entry[4]) if len(entry) >= 5 else self.VELOCITY
                vel = max(1, min(127, vel))
                events.append((t0 - t_start, "on", midi, vel))
                events.append((t1 - t_start, "off", midi, 0))
            events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

            active_notes: set[int] = set()
            if sound is not None:
                sound.play()
            wall_start = time.perf_counter()

            for ev_time, ev_type, note, vel in events:
                target = wall_start + ev_time
                while True:
                    now = time.perf_counter()
                    if now >= target:
                        break
                    remaining = target - now
                    if stop_event.is_set():
                        break
                    time.sleep(min(remaining * 0.5, 0.002))
                if stop_event.is_set():
                    break
                if ev_type == "on":
                    port.note_on(note, vel, channel)
                    active_notes.add(note)
                else:
                    port.note_off(note, 0, channel)
                    active_notes.discard(note)
        finally:
            for note in list(active_notes):
                try:
                    port.note_off(note, 0, channel)
                except Exception:
                    pass
            if sound is not None and stop_event.is_set():
                try:
                    sound.stop()
                except Exception:
                    pass
