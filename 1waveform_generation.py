# waveform_generation.py
# Responsibilities:
#   - Represent waveforms as WaveBundle objects
#   - Generate sine-combination waveforms from USER_main parameters
#   - Generate white/pink/brown noise, custom waveforms
#   - Support mixing selectors (e.g. 'sine1+noise_white')
#   - Build per-channel bundles for different AWG modes

import math
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt

import utilities
import USER_main


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WaveBundle:
    samples: np.ndarray                 # int8, length N
    markers: Optional[np.ndarray] = None  # uint8 bit0=M1, bit1=M2


# ============================================================
# BASIC HELPERS
# ============================================================

def process_noise_only(y, ch):
    """
    Applies noise-only post-processing.
    Sines and other waveforms bypass this completely.
    """
    mode = USER_main.noise_post_process_mode_ch.get(ch, "clip_hard")
    soft_k = USER_main.noise_soft_clip_strength_ch.get(ch, 2.0)

    if mode == "normalize":
        peak = np.max(np.abs(y))
        if peak > 0:
            return y / peak

    elif mode == "clip_hard":
        return np.clip(y, -1.0, 1.0)

    elif mode == "clip_soft":
        return np.tanh(soft_k * y)

    elif mode == "none":
        return y

    else:
        raise ValueError(f"Invalid noise post mode for ch{ch}: {mode}")

    return y

def force_noise_legacy_safe(y: np.ndarray, headroom: float = 0.9) -> np.ndarray:
    """
    Legacy compatibility mode with safety headroom.
    Normalizes noise to full-scale, then applies headroom so
    sine + noise will not hard-clip.
    """
    peak = np.max(np.abs(y))
    if peak > 1e-12:
        y = y / peak
    return headroom * y

def final_dac_stage(y):
    """
    Single final protection + DAC quantization.
    """
    y = np.clip(y, -1.0, 1.0)
    return np.clip(np.round(y * 127), -127, 127).astype(np.int8)

def _auto_normalize(y: np.ndarray) -> np.ndarray:
    """
    Scales waveform to ±1.0 if it exceeds safe range.
    Prevents clipping before DAC conversion.
    """
    max_abs = np.max(np.abs(y))
    if max_abs > 1e-9 and max_abs > 1.0:
        return y / max_abs
    return y


def _apply_crossfade(y: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0:
        return y
    n = len(y)
    fade_samples = min(fade_samples, max(1, n // 4))
    w = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
    y[:fade_samples] *= w              # fade-in
    y[-fade_samples:] *= w[::-1]       # fade-out
    return y


def pack_markers(m1: Optional[np.ndarray], m2: Optional[np.ndarray], N: int) -> Optional[np.ndarray]:
    if m1 is None and m2 is None:
        return None
    m1b = np.zeros(N, dtype=np.uint8) if m1 is None else (m1.astype(np.uint8) & 1)
    m2b = np.zeros(N, dtype=np.uint8) if m2 is None else (m2.astype(np.uint8) & 1)
    return (m1b | (m2b << 1)).astype(np.uint8)


def _quantize_count(N: int, granularity: int, cap: int = int(1e8)) -> int:
    """
    Keysight M8195A-safe quantization.
    SRD = 1 → granularity = 256.
    Minimum segment = granularity.
    """
    # Enforce minimum one granularity block
    if N < granularity:
        N = granularity

    # Round up to nearest granularity block
    Nq = math.ceil(N / granularity) * granularity

    # Enforce maximum
    return min(Nq, cap)



# ============================================================
# SINE-COMBO GENERATION (FROM USER_MAIN)
# ============================================================
def _sine_comb_from_user(params, Fs, N_samples, preview):
    N = int(N_samples)
    t = np.arange(N, dtype=np.float64) / Fs

    y = np.zeros(N, dtype=np.float64)

    for f, ph, a, off in zip(
        params["frequencies"],
        params["phases"],
        params["amplitudes"],
        params["offsets"],
    ):
        y += a * np.sin(2 * np.pi * f * t + ph) + off

    if preview:
        plt.figure(figsize=(7,4))
        plt.plot(y[:2000])
        plt.title("Preview: Sine (fixed N)")
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.001)

    return WaveBundle(samples=y.astype(np.float64))





def _none_bundle(Fs: float, granularity: int) -> WaveBundle:
    N = _quantize_count(math.ceil(Fs * 1e-6), granularity)
    return WaveBundle(samples=np.zeros(N, dtype=np.int8), markers=None)


def generate_custom_wave(Fs: float, granularity: int) -> WaveBundle:
    """
    Placeholder for custom/arbitrary waveform.
    For now, return 0V flat waveform (int8 = 0).
    """
    N = _quantize_count(int(Fs * 1e-6), granularity)  # 1 microsecond flat
    y = np.zeros(N, dtype=float)
    y = _auto_normalize(y)
    y = np.clip(np.round(y * 127), -127, 127).astype(np.int8)
    return WaveBundle(samples=y, markers=None)


# ============================================================
# NOISE GENERATORS
# ============================================================

def _generate_white_noise(N: int) -> np.ndarray:
    """Gaussian white noise."""
    return np.random.normal(0, 1, N)


def _generate_brown_noise(N: int) -> np.ndarray:
    """Brownian (integrated) noise."""
    white = np.random.normal(0, 1, N)
    return np.cumsum(white)


def _generate_pink_noise(N: int) -> np.ndarray:
    """Pink 1/f noise (Voss-McCartney algorithm)."""
    N = int(N)
    white = np.random.normal(0, 1, N)
    freqs = np.fft.rfftfreq(N)
    spectrum = np.fft.rfft(white)
    spectrum[1:] /= np.sqrt(freqs[1:] + 1e-6)  # Avoid divide by zero
    return np.fft.irfft(spectrum, n=N)


# ============================================================
# SINGLE-COMPONENT BUNDLE GENERATION
# ============================================================

def _generate_single_bundle(
    selector: str,
    Fs: float,
    N_samples: int,
    preview: bool,
) -> WaveBundle:

    # ---- SINE WAVEFORMS ----
    if selector.startswith("sine"):
        try:
            # "sine1" → "sine_combination_waveform1"
            func_name = f"sine_combination_waveform{selector[4:]}"
            func = getattr(USER_main, func_name)
        except AttributeError:
            raise ValueError(f"Unknown sine selector: {selector}")

        params = func()
        return _sine_comb_from_user(params, Fs, N_samples, preview)

    # ---- NONE ----
    if selector == "none":
        return WaveBundle(samples=np.zeros(int(N_samples), dtype=np.int8))

    # ---- FUNCn ----
    if selector.startswith("func"):
        y = np.zeros(int(N_samples), dtype=np.float64)
        y = final_dac_stage(y)
        return WaveBundle(samples=y)

    # ---- NOISE ----
    if selector.startswith("noise_"):
        N = int(N_samples)

        if selector == "noise_white":
            y = _generate_white_noise(N) * USER_main.noise_white_amplitude
        elif selector == "noise_pink":
            y = _generate_pink_noise(N) * USER_main.noise_pink_amplitude
        elif selector == "noise_brown":
            y = _generate_brown_noise(N) * USER_main.noise_brown_amplitude
        else:
            raise ValueError(f"Unknown noise selector: {selector}")

        fade = int(USER_main.noise_fade_fraction * N)
        if fade > 0:
            y = _apply_crossfade(y, fade)

        return WaveBundle(samples=y.astype(np.float64))

    raise ValueError(f"Unsupported selector '{selector}'")




# ============================================================
# MIXED-COMPONENT BUNDLE GENERATION
# ============================================================

'''
def _generate_mixed_bundle(
    selector: str,
    Fs: float,
    granularity: int,
    preview: bool,
    segment_length_ms: int
) -> WaveBundle:
    """
    Handle selectors like 'sine1+noise_white+func1'.
    Each component is generated as a single bundle, then summed.
    """
    parts = [s.strip() for s in selector.split('+') if s.strip()]
    if not parts:
        raise ValueError("Empty mixed selector.")

    bundles = []
    for part in parts:
        b = _generate_single_bundle(part, Fs, granularity, False, segment_length_ms)
        bundles.append(b)

    # Use common length = min length of all components
    lengths = [len(b.samples) for b in bundles if b is not None]
    if not lengths:
        raise ValueError(f"No valid components in selector '{selector}'")
    N = min(lengths)

    # Sum as int16 to avoid overflow, then clip to int8
    acc = np.zeros(N, dtype=np.float64)

    for idx, b in enumerate(bundles):
        part = b.samples[:N]

        # Noise-only processing
        if parts[idx].startswith("noise_"):
            part = process_noise_only(part, ch=1)

        acc += part

    
    combined = final_dac_stage(acc)


    if preview:
        try:
            plt.figure()
            plt.plot(combined[:2000])
            plt.title(f"Preview: {selector}  (mixed first 2000 samples)")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude (DAC code)")
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Preview plot failed: {e}")

    return WaveBundle(samples=combined, markers=None)
'''
def generate_wave_bundle(
    selector: str,
    Fs: float,
    N_samples: int,
    preview: bool,
) -> WaveBundle:

    # Mixed: returns FLOAT internally
    if '+' in selector:
        wb = _generate_mixed_bundle(selector, Fs, N_samples, preview)
        wb.samples = final_dac_stage(wb.samples)  # single quantization
        return wb

    # Single: ALWAYS return FLOAT, then quantize once here
    wb = _generate_single_bundle(selector, Fs, N_samples, preview)
    wb.samples = final_dac_stage(wb.samples.astype(np.float64))
    return wb



def _generate_mixed_bundle(
    selector: str,
    Fs: float,
    N_samples: int,
    preview: bool,
) -> WaveBundle:
    """
    Handle selectors like 'sine1+noise_white'.
    All components use the SAME fixed length N_samples.
    """

    parts = [p.strip() for p in selector.split('+') if p.strip()]
    if not parts:
        raise ValueError("Empty mixed selector.")

    N = int(N_samples)
    bundles = [
        _generate_single_bundle(p, Fs, N, False)
        for p in parts
    ]

    acc = np.zeros(N, dtype=np.float64)
    raw_sine = None
    raw_noise = None
    proc_noise = None

    for name, b in zip(parts, bundles):
        x = b.samples.astype(np.float64)

        if name.startswith("noise_"):
            raw_noise = x.copy()
            proc_noise = process_noise_only(x, ch=1)
            proc_noise = force_noise_legacy_safe(proc_noise, headroom=0.9)
            acc += proc_noise
        else:
            raw_sine = x.copy()
            acc += x

    combined = acc.astype(np.float64)


    if preview:
        try:
            plt.figure(figsize=(10, 8))

            if raw_sine is not None:
                plt.subplot(4, 1, 1)
                plt.plot(raw_sine[:2000]); plt.title("RAW SINE"); plt.grid()

            if raw_noise is not None:
                plt.subplot(4, 1, 2)
                plt.plot(raw_noise[:2000]); plt.title("RAW NOISE"); plt.grid()

            if proc_noise is not None:
                plt.subplot(4, 1, 3)
                plt.plot(proc_noise[:2000]); plt.title("PROCESSED NOISE"); plt.grid()

            plt.subplot(4, 1, 4)
            plt.plot(combined[:2000]); plt.title("FINAL DAC OUTPUT"); plt.grid()

            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    return WaveBundle(samples=combined, markers=None)


# ============================================================
# BUNDLE BUILDING PER MODE
# ============================================================

def build_waveform_bundles(
    mode: int,
    ch_outputs: List[str],
    Fs: float,
    granularity: int,
    preview: bool,
    segment_length_ms: int
) -> Dict[str, Optional[WaveBundle]]:

    out: Dict[str, Optional[WaveBundle]] = {'ch1': None, 'ch2': None, 'ch3': None, 'ch4': None}

    # --- ONE universal N_samples for EVERYTHING ---
    duration_s = segment_length_ms * 1e-3
    N_raw = int(Fs * duration_s)
    N_samples = _quantize_count(N_raw, granularity)

    if mode == 1:
        out['ch1'] = generate_wave_bundle(ch_outputs[0], Fs, N_samples, preview)

    elif mode == 2:
        b = generate_wave_bundle(ch_outputs[0], Fs, N_samples, preview)
        b.markers = pack_markers(None, None, len(b.samples))
        out['ch1'] = b

    elif mode == 3:
        out['ch1'] = generate_wave_bundle(ch_outputs[0], Fs, N_samples, preview)
        out['ch4'] = generate_wave_bundle(ch_outputs[3], Fs, N_samples, preview)

    elif mode == 4:
        a = generate_wave_bundle(ch_outputs[0], Fs, N_samples, preview)
        c = generate_wave_bundle(ch_outputs[1], Fs, N_samples, preview)
        out['ch1'] = a
        out['ch3'] = a
        out['ch2'] = c
        out['ch4'] = c

    elif mode == 5:
        a = generate_wave_bundle(ch_outputs[0], Fs, N_samples, preview)
        b = generate_wave_bundle(ch_outputs[1], Fs, N_samples, preview)
        a.markers = pack_markers(None, None, len(a.samples))
        b.markers = pack_markers(None, None, len(b.samples))
        out['ch1'] = a
        out['ch2'] = b

    elif mode == 6:
        out['ch1'] = generate_wave_bundle(ch_outputs[0], Fs, N_samples, preview)
        out['ch2'] = generate_wave_bundle(ch_outputs[1], Fs, N_samples, preview)
        out['ch3'] = generate_wave_bundle(ch_outputs[2], Fs, N_samples, preview)
        out['ch4'] = generate_wave_bundle(ch_outputs[3], Fs, N_samples, preview)

    else:
        raise Exception("Invalid generation mode 1–6.")

    return out



# ============================================================
# LEGACY HELPERS (UNUSED IN USER_MAIN, KEPT FOR COMPAT)
# ============================================================

def sine_comb_parameters(freq, sample_rate, duration, granularity):
    f_max = max(freq)
    if sample_rate < 2 * f_max:
        print("Warning: Sample rate too low for max frequency; may cause aliasing.")
    sample_count = math.ceil(sample_rate * duration)
    sample_count = math.ceil(sample_count / granularity) * granularity
    max_samples = 2147483648
    if sample_count > max_samples:
        print(f"Sample count {sample_count} exceeds maximum {max_samples}. Reducing to maximum.")
        sample_count = max_samples
    return sample_count


def normalize_waveform(waveform):
    w_max = max(waveform)
    w_min = min(waveform)
    if w_max == w_min:
        return np.zeros_like(waveform)
    normalized = (waveform - w_min) / (w_max - w_min)
    normalized = normalized * (127 - (-127)) + (-127)
    return normalized


def generate_sine_combination(freq, phase, amp, off, sample_rate, granularity, plot_preview=True):
    # Legacy, currently unused (not called in USER_main)
    try:
        freq = [float(f) for f in freq]
    except (ValueError, TypeError) as e:
        raise ValueError(f"All frequencies must be numeric; got {freq}") from e

    freq_int = [int(f) for f in freq]
    if any(f != fi for f, fi in zip(freq, freq_int)):
        print("Warning: Non-integer frequencies detected; approximating with integer GCD.")
    gcd_freq = math.gcd(*freq_int) if freq_int else 1
    duration = 1.0 / gcd_freq

    ideal_sample_count = math.ceil(sample_rate * duration)
    if ideal_sample_count > 1e8:
        min_freq = min(freq) if freq else 1.0
        duration = 10 / min_freq
        print("Using fallback duration for memory safety.")

    sample_count = sine_comb_parameters(freq, sample_rate, duration, granularity)
    t = np.linspace(0, sample_count / sample_rate, sample_count, endpoint=False)

    waveform = np.zeros(sample_count)
    for f, p, a, o in zip(freq, phase, amp, off):
        waveform += (a * np.sin(2 * np.pi * f * t + p) + o)

    waveform = waveform * 127
    if np.max(waveform) > 127 or np.min(waveform) < -127:
        print("Warning: amplitude exceeded 1 Vpp; normalizing.")
        waveform = normalize_waveform(waveform)

    waveform = np.round(waveform).astype(np.int8)

    if plot_preview:
        voltage = waveform * (0.5 / 127)
        print(f"Total samples: {sample_count}")
        print(f"Waveform duration: {duration} seconds")
        fig, axs = plt.subplots(2, sharex=True, figsize=(10, 6))
        fig.suptitle(f'Combined Sine Waveform ({sample_count} samples)')
        axs[0].plot(waveform)
        axs[0].set_xlabel("Sample Index")
        axs[0].set_ylabel("Amplitude (DAC)")
        axs[0].grid(True)
        axs[1].plot(voltage)
        axs[1].set_xlabel("Sample Index")
        axs[1].set_ylabel("Amplitude (V)")
        axs[1].grid(True)
        plt.tight_layout()
        plt.show(block=False)

    return waveform
