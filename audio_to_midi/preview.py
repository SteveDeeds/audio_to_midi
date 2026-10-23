"""Preview playback that keeps notes aligned with the stem.

The default path renders notes into the same sample buffer as the audio, so
both reach the speakers through one device with identical delay. The optional
external-MIDI path cannot share a clock with the audio device; it applies a
user-measured latency offset instead.
"""

import threading
import time

import numpy as np


AUDIO_GAINS = (0.25, 1.0)  # stem mostly right
NOTE_GAINS = (1.0, 0.3)  # notes mostly left
ATTACK_SECONDS = 0.005
RELEASE_SECONDS = 0.03
HEADROOM = 0.95
MIDI_PAN_LEFT = 16
SYNC_TEST_PITCH = 76  # GM hi wood block on the drum channel


def render_notes(rows, start_seconds: float, length: int, sample_rate: int, drums: bool = False) -> np.ndarray:
    """Render (start_s, end_s, pitch, velocity) rows into a mono buffer of `length` samples.

    Sample 0 corresponds to `start_seconds`, the same instant as the stem segment's first sample.
    """
    out = np.zeros(length, dtype=np.float32)
    for start, end, pitch, velocity in rows:
        first = max(0, round((start - start_seconds) * sample_rate))
        last = min(length, round((end - start_seconds) * sample_rate))
        if drums:
            last = min(length, first + round(0.12 * sample_rate))
        if last <= first:
            continue
        t = np.arange(last - first, dtype=np.float32) / sample_rate
        amp = 0.3 * max(1, min(127, int(velocity))) / 127
        if drums:
            out[first:last] += amp * _drum(int(pitch), t)
            continue
        freq = 440.0 * 2 ** ((int(pitch) - 69) / 12)
        tone = sum(weight * np.sin(2 * np.pi * freq * k * t) for k, weight in ((1, 1.0), (2, 0.3), (3, 0.15)))
        duration = t[-1] + 1 / sample_rate
        envelope = np.minimum(1, np.minimum(t / ATTACK_SECONDS, (duration - t) / RELEASE_SECONDS))
        out[first:last] += amp * tone * np.clip(envelope, 0, 1)
    return out


def _drum(pitch: int, t: np.ndarray) -> np.ndarray:
    if pitch in (35, 36):
        return np.sin(2 * np.pi * (50 * t + 60 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 25)
    noise = np.random.default_rng(pitch).uniform(-1, 1, len(t)).astype(np.float32)
    return noise * np.exp(-t * (60 if pitch >= 42 else 30))


def click_track(times, length: int, sample_rate: int, frequency: float = 1000.0, gain: float = 1.0) -> np.ndarray:
    """Short clicks at `times` (seconds from sample 0)."""
    out = np.zeros(length, dtype=np.float32)
    t = np.arange(round(0.02 * sample_rate), dtype=np.float32) / sample_rate
    click = gain * np.sin(2 * np.pi * frequency * t) * np.exp(-t * 200)
    for when in times:
        first = round(when * sample_rate)
        if 0 <= first < length:
            part = click[: length - first]
            out[first:first + len(part)] += part
    return out


def mix(audio: np.ndarray, notes: np.ndarray | None = None) -> np.ndarray:
    """Return int16 stereo: stem mostly right and notes mostly left, or a centered stem alone."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    audio = audio / peak * 0.8 if peak > 1e-6 else np.zeros_like(audio)
    if notes is None:
        stereo = np.column_stack([audio, audio])
    else:
        notes = np.asarray(notes, dtype=np.float32)[: len(audio)]
        notes = np.pad(notes, (0, len(audio) - len(notes)))
        stereo = np.column_stack([
            audio * AUDIO_GAINS[0] + notes * NOTE_GAINS[0],
            audio * AUDIO_GAINS[1] + notes * NOTE_GAINS[1],
        ])
    loudest = float(np.max(np.abs(stereo))) if len(stereo) else 0.0
    if loudest > HEADROOM:
        stereo *= HEADROOM / loudest
    return (stereo * 32767).astype(np.int16)


def midi_schedule(notes, latency_seconds: float) -> list[tuple[float, str, int, int]]:
    """Wall-clock send times relative to playback start, plus the audio start.

    `latency_seconds` is how much later external MIDI is heard than audio sent at
    the same moment. Positive values send MIDI early (delaying audio if needed).
    """
    lead = max(0.0, latency_seconds)
    events = [(lead, "audio", 0, 0)]
    for start, end, pitch, velocity in notes:
        events.append((lead + start - latency_seconds, "on", int(pitch), max(1, min(127, int(velocity)))))
        events.append((lead + end - latency_seconds, "off", int(pitch), 0))
    order = {"off": 0, "audio": 1, "on": 2}
    return sorted(((max(0.0, t), kind, p, v) for t, kind, p, v in events), key=lambda e: (e[0], order[e[1]]))


def midi_outputs() -> list[str]:
    import pygame.midi

    if not pygame.midi.get_init():
        pygame.midi.init()
    names = []
    for index in range(pygame.midi.get_count()):
        _, name, _, is_output, _ = pygame.midi.get_device_info(index)
        if is_output:
            names.append(name.decode(errors="replace"))
    return names


class Preview:
    """pygame-backed player; one preview at a time."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._port = None
        self._port_name: str | None = None

    def _sound(self, stereo: np.ndarray, sample_rate: int):
        import pygame.mixer
        import pygame.sndarray

        current = pygame.mixer.get_init()
        if current and (current[0] != sample_rate or current[2] != 2):
            pygame.mixer.quit()
            current = None
        if not current:
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=1024)
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

    def play_audio(self, stereo: np.ndarray, sample_rate: int) -> None:
        self.stop()
        self._sound(stereo, sample_rate).play()

    def play_with_midi(self, stereo: np.ndarray, sample_rate: int, notes, device: str,
                       channel: int, latency_seconds: float) -> None:
        """Play the stem and send `notes` (seconds from the segment start) to `device`."""
        self.stop()
        port = self._open(device)
        sound = self._sound(stereo, sample_rate)
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=self._run, args=(port, sound, notes, channel, latency_seconds, stop_event),
            daemon=True, name="midi-preview",
        )
        self._thread.start()

    def _open(self, device: str):
        import pygame.midi

        if self._port is not None and self._port_name == device:
            return self._port
        self.close_port()
        if not pygame.midi.get_init():
            pygame.midi.init()
        for index in range(pygame.midi.get_count()):
            _, name, _, is_output, _ = pygame.midi.get_device_info(index)
            if is_output and name.decode(errors="replace") == device:
                self._port = pygame.midi.Output(index, latency=0)
                self._port_name = device
                return self._port
        raise ValueError(f"MIDI output {device!r} is not available.")

    @staticmethod
    def _run(port, sound, notes, channel, latency_seconds, stop_event) -> None:
        active: set[int] = set()
        try:
            port.write_short(0xB0 | channel, 10, MIDI_PAN_LEFT)
            port.write_short(0xB0 | channel, 7, 127)
            begin = time.perf_counter()
            for when, kind, pitch, velocity in midi_schedule(notes, latency_seconds):
                while not stop_event.is_set():
                    remaining = begin + when - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining * 0.5, 0.002))
                if stop_event.is_set():
                    break
                if kind == "audio":
                    sound.play()
                elif kind == "on":
                    port.note_on(pitch, velocity, channel)
                    active.add(pitch)
                else:
                    port.note_off(pitch, 0, channel)
                    active.discard(pitch)
        finally:
            for pitch in active:
                port.note_off(pitch, 0, channel)

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._thread = None
        self._stop_event = None
        import pygame.mixer

        if pygame.mixer.get_init():
            pygame.mixer.stop()

    def close_port(self) -> None:
        if self._port is not None:
            self._port.close()
        self._port = None
        self._port_name = None
