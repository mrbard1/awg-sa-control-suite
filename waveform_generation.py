# waveform_generation.py
# Responsibilities:
#   - Represent waveforms as WaveBundle objects
#   - Generate sine-combination waveforms from USER_main parameters
#   - Generate white/pink/brown noise, custom waveforms
#   - Support mixing selectors (e.g. "sine1+noise_white")
#   - Build per-channel bundles for different AWG modes
#
# Design:
#   - Pure sines and funcN waveforms use "natural" lengths, not segment_length_ms.
#   - Noise and any mix including noise use segment_length_ms (with a hard cap).
#   - All outputs are int8 DAC codes in [-127, 127].

import math
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import __main__ as USER_main
except Exception:
    import USER_main


# data structure

@dataclass
class WaveBundle:
    samples: np.ndarray                  # np.int8 after final DAC stage
    markers: Optional[np.ndarray] = None  # optional uint8 marker bits
    awg: Optional[dict] = None


# helpers

def pack_markers(m1: Optional[np.ndarray], m2: Optional[np.ndarray], N: int) -> Optional[np.ndarray]:
    """
    Pack two boolean/0-1 marker arrays into a single uint8 bitfield:
      bit0 -> M1, bit1 -> M2.
    """
    if m1 is None and m2 is None:
        return None
    m1b = np.zeros(N, dtype=np.uint8) if m1 is None else (np.asarray(m1, dtype=np.uint8) & 1)
    m2b = np.zeros(N, dtype=np.uint8) if m2 is None else (np.asarray(m2, dtype=np.uint8) & 1)
    return (m1b | (m2b << 1)).astype(np.uint8)


def _quantize_count(N: int, granularity: int, cap: int = int(16e6)) -> int:
    """
    Quantize N up to the next multiple of 'granularity', enforcing:
      - at least one granularity block
      - a maximum 'cap' samples (default 16M to avoid huge segments).
    """
    if N < granularity:
        N = granularity
    Nq = int(math.ceil(N / granularity) * granularity)
    if Nq > cap:
        Nq = cap
    return Nq


def _compute_segment_length(Fs: float, granularity: int, segment_length_ms: int, cap: Optional[int] = None) -> int:
    """
    Compute a legal segment length (in samples) for the given sample rate,
    granularity, and desired duration in ms. Used for noise / noise mixes.
    """
    duration_s = segment_length_ms * 1e-3
    N_raw = int(Fs * duration_s)
    if cap is None:
        return _quantize_count(N_raw, granularity)
    return _quantize_count(N_raw, granularity, cap=cap)

def _get_awg_settings_for_selector(selector: str) -> dict:
    if selector.startswith("sine"):
        func_name = f"sine_combination_waveform{selector[4:]}"
        func = getattr(USER_main, func_name)
        params = func()
        return params.get("awg", {}) or {}
    return {}

def _volts_to_norm(y_volts: np.ndarray, vpp: float) -> np.ndarray:
    vpk_fs = max(float(vpp) / 2.0, 1e-12)
    return np.asarray(y_volts, dtype=np.float64) / vpk_fs



def _apply_crossfade(y: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Apply a simple linear fade-in/fade-out window of length fade_samples
    at the start and end of the waveform to reduce discontinuities.
    """
    if fade_samples <= 0:
        return y
    n = len(y)
    fade_samples = min(fade_samples, max(1, n // 4))
    w = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
    y = y.copy()
    y[:fade_samples] *= w           # fade-in
    y[-fade_samples:] *= w[::-1]    # fade-out
    return y


def final_dac_stage(y: np.ndarray) -> np.ndarray:
    """
    FINAL and ONLY place where float → int8 DAC codes happen.

    Steps:
    - ensure float64
    - remove NaNs/infs
    - clip to [-1, +1]
    - scale by 127
    - round and cast to int8
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.zeros(0, dtype=np.int8)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1.0, 1.0)

    codes = np.round(y * 127.0)
    codes = np.clip(codes, -127, 127).astype(np.int8)
    return codes


# Noise

def _generate_white_noise(N: int) -> np.ndarray:
    return np.random.randn(N).astype(np.float64)


def _generate_brown_noise(N: int) -> np.ndarray:
    w = np.random.randn(N).astype(np.float64)
    y = np.cumsum(w)
    peak = np.max(np.abs(y))
    if peak > 1e-12:
        y /= peak
    return y


def _generate_pink_noise(N: int) -> np.ndarray:
    """
    Voss-McCartney pink noise generator.
    Produces true 1/f noise with -3 dB/octave spectrum.

    Efficient O(N) implementation suitable for large N, unlike the naive matrix version.
    """

    # Number of random sources
    num_sources = 16

    # Running sum of all random rows
    total = np.zeros(N, dtype=np.float64)


    rows = np.random.randn(num_sources)

    counters = np.zeros(num_sources, dtype=np.int64)

    for i in range(N):
       
        k = np.random.randint(num_sources)

        
        rows[k] = np.random.randn()
        counters[k] += 1


        total[i] = rows.sum()

    # Normalize to +-1
    peak = np.max(np.abs(total))
    if peak > 1e-12:
        total /= peak

    return total



def process_noise_only(y: np.ndarray, ch: int) -> np.ndarray:
    """
    Apply per-channel noise post-processing based on USER_main settings:
      - noise_post_process_mode_ch: "none", "normalize", "clip_hard", "clip_soft"
      - noise_soft_clip_strength_ch: float, used by "clip_soft"
    """
    mode = USER_main.noise_post_process_mode_ch.get(ch, "none")
    k = USER_main.noise_soft_clip_strength_ch.get(ch, 2.0)

    y = np.asarray(y, dtype=np.float64)

    if mode == "none":
        return y

    max_abs = np.max(np.abs(y)) if y.size > 0 else 0.0

    if mode == "normalize":
        if max_abs > 1e-12:
            return y / max_abs
        return y

    if mode == "clip_hard":
        return np.clip(y, -1.0, 1.0)

    if mode == "clip_soft":
        if k <= 0:
            return np.clip(y, -1.0, 1.0)
        return np.tanh(k * y) / np.tanh(k)

    return y


def force_noise_legacy_safe(y: np.ndarray, headroom: float = 0.9) -> np.ndarray:
    """
    Legacy compatibility mode with safety headroom.
    Normalizes noise to full-scale, then applies headroom so
    sine + noise will not hard-clip too aggressively.
    """
    y = np.asarray(y, dtype=np.float64)
    peak = np.max(np.abs(y)) if y.size > 0 else 0.0
    if peak > 1e-12:
        y = y / peak
    return headroom * y


# sine generation from main

def _sine_comb_from_user(
    params: Dict,
    Fs: float,
    N_samples: int,
    preview: bool,
) -> np.ndarray:
    """
    Generate a sum of sines (and DC offsets) using EXACTLY N_samples.
    Returns a float64 array; DAC conversion happens later.
    """
    N = int(N_samples)
    t = np.arange(N, dtype=np.float64) / Fs

    y = np.zeros(N, dtype=np.float64)
    freqs      = params.get("frequencies", [])
    phases     = params.get("phases", [])
    amplitudes = params.get("amplitudes", [])
    offsets    = params.get("offsets", [])

    for f, ph, a, off in zip(freqs, phases, amplitudes, offsets):
        y += a * np.sin(2.0 * np.pi * f * t + ph) + off

    if preview:
        try:
            plt.figure(figsize=(7, 4))
            plt.plot(y[:2000])
            plt.title("Preview: Sine (float)")
            plt.grid(True)
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass

    return y


def _sine_segment_length_from_params(
    params: Dict,
    Fs: float,
    granularity: int,
    max_samples: int = int(4e6),
) -> int:
    """
    Decide a "natural" segment length for a sine combination:
      - Try to include ~200 cycles of the lowest non-zero frequency.
      - Cap at max_samples to avoid huge segments.
    """
    freqs = [f for f in params.get("frequencies", []) if f > 0]
    if not freqs:
        # No frequencies defined → default to 1 µs of data.
        duration_s = 1e-6
    else:
        f_min = min(freqs)
        cycles = 200.0
        duration_s = cycles / f_min

    N_raw = int(Fs * duration_s)
    if N_raw <= 0:
        N_raw = int(Fs * 1e-6)
    return _quantize_count(N_raw, granularity, cap=max_samples)


# single bundle

def _generate_single_bundle(
    selector: str,
    Fs: float,
    granularity: int,
    preview: bool,
    segment_length_ms: int,
    ch_index: int = 1,
) -> WaveBundle:
    """
    Generate a single-component waveform (sine, noise, func, none),
    returning a WaveBundle with FLOAT samples.
    DAC conversion happens later in generate_wave_bundle().
    """

    # empty array
    if selector == "none":
        N = granularity
        return WaveBundle(samples=np.zeros(N, dtype=np.float64))

    # sines
    if selector.startswith("sine"):
        try:
            func_name = f"sine_combination_waveform{selector[4:]}"
            func = getattr(USER_main, func_name)
        except AttributeError:
            raise ValueError(f"Unknown sine selector: {selector}")

        params = func()
        N = _sine_segment_length_from_params(params, Fs, granularity)
        y = _sine_comb_from_user(params, Fs, N, preview)
        return WaveBundle(samples=y, awg=params.get("awg", None))


    # custom user function
    if selector.startswith("func"):
        idx = selector[4:]
        func_name = f"user_waveform{idx}" if idx else "user_waveform1"
        func = getattr(USER_main, func_name, None)

        if func is None:
            # zero waveform
            N = granularity
            y_long = np.zeros(N, dtype=np.float64)
        else:
            y = np.asarray(func(), dtype=np.float64)
            if y.ndim == 0 or y.size == 0:
                y = np.zeros(1, dtype=np.float64)

            # quantize to legal length to avoid awg crash
            N = _quantize_count(len(y), granularity, cap=int(4e6))
            if N <= len(y):
                y_long = y[:N]
            else:
                reps = int(math.ceil(N / len(y)))
                y_long = np.tile(y, reps)[:N]

        if preview:
            try:
                plt.figure(figsize=(7, 4))
                plt.plot(y_long[:2000])
                plt.title(f"Preview: {selector} (func)")
                plt.grid(True)
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass

        return WaveBundle(samples=y_long, awg=params.get("awg", None))


    # noise only
    if selector.startswith("noise_"):
        # Use segment_length_ms, cap to upload reasonable waveform
        N = _compute_segment_length(Fs, granularity, segment_length_ms, cap=int(16e6))

        if selector == "noise_white":
            y = _generate_white_noise(N) * USER_main.noise_white_amplitude
        elif selector == "noise_pink":
            y = _generate_pink_noise(N) * USER_main.noise_pink_amplitude
        elif selector == "noise_brown":
            y = _generate_brown_noise(N) * USER_main.noise_brown_amplitude
        else:
            raise ValueError(f"Unknown noise selector: {selector}")

        fade_frac = getattr(USER_main, "noise_fade_fraction", 0.0)
        fade = int(fade_frac * N)
        if fade > 0:
            y = _apply_crossfade(y, fade)

        y_proc = process_noise_only(y, ch_index)

        if preview:
            try:
                plt.figure(figsize=(10, 6))
                plt.subplot(2, 1, 1)
                plt.plot(y[:2000])
                plt.title(f"{selector}: raw")
                plt.grid(True)

                plt.subplot(2, 1, 2)
                plt.plot(y_proc[:2000])
                plt.title(f"{selector}: processed (ch {ch_index})")
                plt.grid(True)

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass

        return WaveBundle(samples=y_proc, awg=params.get("awg", None))

    # unkown waveform selector
    raise ValueError(f"Unsupported selector '{selector}'")


# mixed bundle, superimposed waveforms
def _generate_mixed_bundle(
    selector: str,
    Fs: float,
    granularity: int,
    preview: bool,
    segment_length_ms: int,
    ch_index: int = 1,
) -> WaveBundle:
    """
    Handle selectors like "sine1+noise_white".
    Rules:
      - If ANY noise_* component is present → use segment_length_ms (capped).
      - Otherwise (pure sines/funcs) → use natural lengths and tile/trim.
    Returns FLOAT samples; DAC conversion is applied later.
    """
    parts = [p.strip() for p in selector.split('+') if p.strip()]
    if not parts:
        raise ValueError("Empty mixed selector.")

    

    has_noise = any(p.startswith("noise_") for p in parts)

    # mixture includes noise and real waveform, use segment_length_ms to determine N
    if has_noise:
        N = _compute_segment_length(Fs, granularity, segment_length_ms, cap=int(16e6))
        acc = np.zeros(N, dtype=np.float64)

        raw_sine = None
        raw_noise = None
        proc_noise = None

        for name in parts:
            gain = getattr(USER_main, "component_gains", {}).get(name, 1.0)
            if name.startswith("noise_"):
                if name == "noise_white":
                    y = _generate_white_noise(N) * USER_main.noise_white_amplitude
                elif name == "noise_pink":
                    y = _generate_pink_noise(N) * USER_main.noise_pink_amplitude
                elif name == "noise_brown":
                    y = _generate_brown_noise(N) * USER_main.noise_brown_amplitude
                else:
                    raise ValueError(f"Unknown noise selector in mix: {name}")

                fade_frac = getattr(USER_main, "noise_fade_fraction", 0.0)
                fade = int(fade_frac * N)
                if fade > 0:
                    y = _apply_crossfade(y, fade)

                raw_noise = y.copy()
                proc_noise = process_noise_only(y, ch_index)
                #proc_noise = force_noise_legacy_safe(proc_noise, headroom=0.9)
                
                acc += gain * proc_noise
            else:
                single = _generate_single_bundle(name, Fs, granularity, False, segment_length_ms, ch_index)
                x = np.asarray(single.samples, dtype=np.float64)
                if x.size == 0:
                    continue
                if len(x) < N:
                    reps = int(math.ceil(N / len(x)))
                    x_long = np.tile(x, reps)[:N]
                else:
                    x_long = x[:N]
                raw_sine = x_long.copy()
                acc += gain * x_long
        
        vpp = 1.0
        for name in parts:
            if name.startswith("sine"):
                awg = _get_awg_settings_for_selector(name)
                vpp = float(awg.get("vpp", vpp))
                break


        
        headroom = getattr(USER_main, "mix_headroom", 1.0)
        acc *= headroom
        if preview:
            try:
                plt.figure(figsize=(10, 8))
                idx = 1
                if raw_sine is not None:
                    plt.subplot(4, 1, idx); idx += 1
                    plt.plot(raw_sine[:2000]); plt.title("RAW SINE"); plt.grid()
                if raw_noise is not None:
                    plt.subplot(4, 1, idx); idx += 1
                    plt.plot(raw_noise[:2000]); plt.title("RAW NOISE"); plt.grid()
                if proc_noise is not None:
                    plt.subplot(4, 1, idx); idx += 1
                    plt.plot(proc_noise[:2000]); plt.title("PROCESSED NOISE"); plt.grid()
                plt.subplot(4, 1, idx)
                plt.plot(acc[:2000]); plt.title("COMBINED FLOAT"); plt.grid()
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass

        return WaveBundle(samples=acc, awg={"vpp": vpp})

    # pure sines
    bundles = []
    for name in parts:
        wb = _generate_single_bundle(name, Fs, granularity, False, segment_length_ms, ch_index)
        x = np.asarray(wb.samples, dtype=np.float64)
        if x.size > 0:
            bundles.append(x)

    if not bundles:
        return WaveBundle(samples=np.zeros(granularity, dtype=np.float64))

    N = max(len(x) for x in bundles)
    N = _quantize_count(N, granularity, cap=int(4e6))

    acc = np.zeros(N, dtype=np.float64)
    for x in bundles:
        if len(x) < N:
            reps = int(math.ceil(N / len(x)))
            x_long = np.tile(x, reps)[:N]
        else:
            x_long = x[:N]
        acc += x_long

    if preview:
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(acc[:2000])
            plt.title("COMBINED SINE/FUNC MIX")
            plt.grid(True)
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            pass

    return WaveBundle(samples=acc)


# single wavebundle, 8 bit int

def generate_wave_bundle(
    selector: str,
    Fs: float,
    granularity: int,
    preview: bool,
    segment_length_ms: int,
    ch_index: int = 1,
) -> WaveBundle:
    """
    Public entry point used by USER_main and noise_streaming.
    Returns a WaveBundle whose samples are INT8 DAC codes.
    """
    if '+' in selector:
        wb = _generate_mixed_bundle(selector, Fs, granularity, preview, segment_length_ms, ch_index)
        awg = {}
    else:
        wb = _generate_single_bundle(selector, Fs, granularity, preview, segment_length_ms, ch_index)
        awg = _get_awg_settings_for_selector(selector)

    vpp = float(awg.get("vpp", 1.0))
    wb.samples = final_dac_stage(_volts_to_norm(wb.samples, vpp))
    return wb


# build per channel bundles 

def build_waveform_bundles(
    mode: int,
    ch_outputs: List[str],
    Fs: float,
    granularity: int,
    preview: bool,
    segment_length_ms: int,
) -> Dict[str, Optional[WaveBundle]]:
    """
    Build a dict of WaveBundle objects for channels ch1..ch4
    according to the chosen "mode" (1..6).

    For your current lab setup, generation_mode = 1, so only mode 1
    is actively used. The other modes are kept for completeness and
    will also obey the same length and type guarantees.
    """

    bundles: Dict[str, Optional[WaveBundle]] = {
        'ch1': None,
        'ch2': None,
        'ch3': None,
        'ch4': None,
    }

    def sel(idx: int) -> str:
        if idx < 0 or idx >= len(ch_outputs):
            return "none"
        return ch_outputs[idx]

    if mode == 1:
        # Single channel (Ch.1)
        bundles['ch1'] = generate_wave_bundle(sel(0), Fs, granularity, preview, segment_length_ms, ch_index=1)

    elif mode == 2:
        # Single channel with markers (Ch.1 – markers not heavily used here)
        b = generate_wave_bundle(sel(0), Fs, granularity, preview, segment_length_ms, ch_index=1)
        N = b.samples.size
        m1 = np.ones(N, dtype=np.uint8)
        b.markers = pack_markers(m1, None, N)
        bundles['ch1'] = b

    elif mode == 3:
        # Dual channel (Ch.1 and Ch.4)
        bundles['ch1'] = generate_wave_bundle(sel(0), Fs, granularity, preview, segment_length_ms, ch_index=1)
        bundles['ch4'] = generate_wave_bundle(sel(3), Fs, granularity, preview, segment_length_ms, ch_index=4)

    elif mode == 4:
        # Dual channel duplicate (Ch.1 = Ch.3, Ch.2 = Ch.4)
        a = generate_wave_bundle(sel(0), Fs, granularity, preview, segment_length_ms, ch_index=1)
        c = generate_wave_bundle(sel(1), Fs, granularity, preview, segment_length_ms, ch_index=2)
        bundles['ch1'] = a
        bundles['ch3'] = a
        bundles['ch2'] = c
        bundles['ch4'] = c

    elif mode == 5:
        # Dual channel with markers (Ch.1 and Ch.2)
        a = generate_wave_bundle(sel(0), Fs, granularity, preview, segment_length_ms, ch_index=1)
        b = generate_wave_bundle(sel(1), Fs, granularity, preview, segment_length_ms, ch_index=2)
        Na = a.samples.size
        Nb = b.samples.size
        a.markers = pack_markers(np.ones(Na, dtype=np.uint8), None, Na)
        b.markers = pack_markers(np.ones(Nb, dtype=np.uint8), None, Nb)
        bundles['ch1'] = a
        bundles['ch2'] = b

    elif mode == 6:
        # Four channel (Ch.1..Ch.4)
        bundles['ch1'] = generate_wave_bundle(sel(0), Fs, granularity, preview, segment_length_ms, ch_index=1)
        bundles['ch2'] = generate_wave_bundle(sel(1), Fs, granularity, preview, segment_length_ms, ch_index=2)
        bundles['ch3'] = generate_wave_bundle(sel(2), Fs, granularity, preview, segment_length_ms, ch_index=3)
        bundles['ch4'] = generate_wave_bundle(sel(3), Fs, granularity, preview, segment_length_ms, ch_index=4)

    else:
        raise ValueError("Invalid generation mode (must be 1..6)")

    return bundles
