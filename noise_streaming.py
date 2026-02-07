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
        # Fixed segment length in samples for this streamer instance.
        # IMPORTANT: static base and noise must be generated at the same N to avoid boundary discontinuities.
        self.noise_len = waveform_generation._compute_segment_length(
            self.Fs,
            self.granularity,
            self.segment_length_ms,
            cap=int(16e6),
        )


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
        self.static_awg = {}
        for ch_num, static_selector, _ in self.noise_channels:
            if static_selector:
                # If static base is a pure sine, force it to the SAME N as the noise segment.
                # Otherwise, sine may be generated at a "natural" N and later trimmed, reintroducing discontinuities.
                if static_selector.startswith("sine") and "+" not in static_selector:
                    b = waveform_generation._generate_sine_bundle_with_N(
                        static_selector,
                        self.Fs,
                        self.noise_len,
                        preview=False,
                        params=None,
                    )
                    awg = waveform_generation._get_awg_settings_for_selector(static_selector)
                    if b.awg:
                        merged = dict(awg or {})
                        merged.update(b.awg)
                        awg = merged
                    vpp = float((awg or {}).get("vpp", 1.0))
                    b.samples = waveform_generation.final_dac_stage(
                        waveform_generation._volts_to_norm(b.samples, vpp)
                    )
                else:
                    # For non-sine or mixed static selectors, use the normal generation path.
                    b = waveform_generation.generate_wave_bundle(
                        static_selector,
                        self.Fs,
                        self.granularity,
                        preview=False,
                        segment_length_ms=self.segment_length_ms,
                        ch_index=ch_num,
                    )

                self.static_base[ch_num] = np.asarray(b.samples, dtype=np.int8)
                self.static_awg[ch_num] = dict(b.awg or {})

            else:
                self.static_base[ch_num] = None
                self.static_awg[ch_num] = {}

        self._stop = threading.Event()
        self._thread = None
        self._active_seg = {ch: 1 for ch, _, _ in self.noise_channels}
        self._last_vpp = {}
        self._last_clip = {}

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
                noise_volts = []
                noise_len = self.noise_len
                gains = getattr(waveform_generation.USER_main, "component_gains", {})

                for nc in noise_comps:
                    yv, _raw, _meta = waveform_generation._generate_noise_volts(nc, noise_len, self.Fs, ch_num)
                    gain = float(gains.get(nc, 1.0))
                    noise_volts.append(gain * np.asarray(yv, dtype=np.float64))


                static_arr = self.static_base.get(ch_num)
                N = noise_len

                # Safety: static base should have been precomputed at exactly N.
                if static_arr is not None and len(static_arr) != N:
                    print(
                        f"NoiseStreamer ch{ch_num}: static base length {len(static_arr)} != noise_len {N}; "
                        "using zeros for base to avoid discontinuity."
                    )
                    static_arr = None


                noise_awg = {}
                for nc in noise_comps:
                    awg_cfg = waveform_generation._get_awg_settings_for_selector(nc)
                    if "vpp" in awg_cfg:
                        noise_awg = awg_cfg
                        break
                base_awg = self.static_awg.get(ch_num, {})
                vpp = float(noise_awg.get("vpp", base_awg.get("vpp", 1.0)))
                if self._last_vpp.get(ch_num) != vpp:
                    print(f"NoiseStreamer ch{ch_num} mix vpp={vpp}")
                    self._last_vpp[ch_num] = vpp

                if static_arr is not None:
                    base_norm = waveform_generation._int8_to_norm(static_arr[:N])
                    base_volts = waveform_generation._norm_to_volts(base_norm, vpp)
                else:
                    base_volts = np.zeros(N, dtype=np.float64)

                acc_volts = base_volts
                for nb in noise_volts:
                    acc_volts = acc_volts + nb[:N]

                headroom = getattr(waveform_generation.USER_main, "mix_headroom", 1.0)
                acc_volts = acc_volts * headroom

                mixed_norm = waveform_generation._volts_to_norm(acc_volts, vpp)
                clip_pct = float(np.mean(np.abs(mixed_norm) > 1.0) * 100.0)
                if clip_pct > 0.0 and self._last_clip.get(ch_num, 0.0) == 0.0:
                    print(f"NoiseStreamer ch{ch_num} clip {clip_pct:.2f}%")
                self._last_clip[ch_num] = clip_pct

                combined = waveform_generation.final_dac_stage(mixed_norm)

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
