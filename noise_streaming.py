# noise_streaming.py
# Implements continuous double-buffer noise streaming (plus mixing
# with static components) for channels whose selector contains 'noise_*'.

import threading
import time
from typing import List, Tuple

import numpy as np

import waveform_generation
import awg_control


class NoiseStreamer:
    """
    Continuously regenerates random noise and updates AWG segments
    for channels whose selector contains any 'noise_*' component.
    Non-noise components (sine, func, etc.) are treated as static base.
    """

    def __init__(
        self,
        awg,
        mode: int,
        channel_outputs: List[str],
        Fs: float,
        granularity: int,
        segment_length_ms: int,
        refresh_factor: float = 0.9,
    ):
        self.awg = awg
        self.mode = mode
        self.ch_outputs = channel_outputs
        self.Fs = Fs
        self.granularity = granularity
        self.segment_length_ms = segment_length_ms
        self.refresh_factor = refresh_factor

        mode_to_channels = {
            1: [1],
            2: [1],
            3: [1, 4],
            4: [1, 2, 3, 4],
            5: [1, 2],
            6: [1, 2, 3, 4],
        }
        self.ch_nums = mode_to_channels.get(mode, [])

        # For each physical channel -> (static_selector or None, [noise_components])
        self.noise_channels: List[Tuple[int, str, List[str]]] = []
        for idx, sel in enumerate(self.ch_outputs):
            if idx >= len(self.ch_nums):
                continue
            ch_num = self.ch_nums[idx]
            parts = [s.strip() for s in sel.split('+') if s.strip()]

            noise_comps = [p for p in parts if p.startswith("noise_")]
            static_parts = [p for p in parts if not p.startswith("noise_")]
            static_selector = "+".join(static_parts)

            if noise_comps:
                self.noise_channels.append((ch_num, static_selector, noise_comps))

        # Precompute static base per channel (if any)
        self.static_base = {}
        for ch_num, static_selector, _ in self.noise_channels:
            if static_selector:
                b = waveform_generation.generate_wave_bundle(
                    static_selector,
                    self.Fs,
                    self.granularity,
                    preview=False,
                    segment_length_ms=self.segment_length_ms,
                )
                self.static_base[ch_num] = np.asarray(b.samples, dtype=np.int8)
            else:
                self.static_base[ch_num] = None

        self._stop = threading.Event()
        self._thread = None
        self._active_seg = {ch: 1 for ch, _, _ in self.noise_channels}

    def _segment_duration(self) -> float:
        return self.segment_length_ms / 1000.0

    def _loop(self):
        seg_dt = self._segment_duration()
        sleep_time = seg_dt * self.refresh_factor

        while not self._stop.is_set():
            for ch_num, _static_selector, noise_comps in self.noise_channels:
                # Ping-pong segment index
                current_seg = self._active_seg[ch_num]
                next_seg = 2 if current_seg == 1 else 1

                # Generate all noise components
                noise_bundles = []
                for nc in noise_comps:
                    nb = waveform_generation.generate_wave_bundle(
                        nc,
                        self.Fs,
                        self.granularity,
                        preview=False,
                        segment_length_ms=self.segment_length_ms,
                    )
                    noise_bundles.append(np.asarray(nb.samples, dtype=np.int8))

                # Decide common length (min of all available arrays)
                lengths = [len(nb) for nb in noise_bundles]
                static_arr = self.static_base.get(ch_num)
                if static_arr is not None:
                    lengths.append(len(static_arr))
                if not lengths:
                    continue
                N = min(lengths)

                acc = np.zeros(N, dtype=np.int16)
                if static_arr is not None:
                    acc += static_arr[:N].astype(np.int16)
                for nb in noise_bundles:
                    acc += nb[:N].astype(np.int16)

                combined = np.clip(acc, -127, 127).astype(np.int8)

                awg_control.update_segment_samples(
                    self.awg,
                    ch_num=ch_num,
                    seg_num=next_seg,
                    samples=combined,
                )
                awg_control.select_segment(self.awg, ch_num, next_seg)
                self._active_seg[ch_num] = next_seg

            time.sleep(sleep_time)

    def start(self):
        if not self.noise_channels:
            print("NoiseStreamer: no channels with noise_* components; nothing to stream.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"NoiseStreamer started for channels: {self.noise_channels}")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        print("NoiseStreamer stopped.")
