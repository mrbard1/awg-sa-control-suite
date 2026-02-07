# AUTO_noise_spectroscopy_test.py
#
# Drop this file into your working directory (same folder as USER_main.py).
# Run:
#   python AUTO_noise_spectroscopy_test.py
#
# Or customize:
#   python AUTO_noise_spectroscopy_test.py --rbw 100e3 10e3 --dens 3e-7 6e-7 1.2e-6 --plot 0

import argparse
import datetime
import os
import numpy as np
import pyvisa

import USER_main
import utilities
import waveform_generation
import awg_control
import sa_control
# ---- globals waveform_generation expects (it imports __main__ as USER_main) ----
noise_white_amplitude = 0.01
noise_pink_amplitude = 0.5
noise_brown_amplitude = 0.5

noise_density_vrthz = None     # we will set per test
noise_seed = 12345             # we will set from args
noise_seed_current = None      # optional, keep None
noise_post_process_mode_ch = {1: "none", 2: "none", 3: "none", 4: "none"}
noise_soft_clip_strength_ch = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def median_dbmhz(freq_hz: np.ndarray, psd_dbmhz: np.ndarray, fmin: float, fmax: float) -> float:
    m = (freq_hz >= fmin) & (freq_hz <= fmax) & np.isfinite(psd_dbmhz)
    if not np.any(m):
        return float("nan")
    return float(np.median(psd_dbmhz[m]))


def write_trace_csv(path: str, freq_hz: np.ndarray, p_dbm: np.ndarray, rbw_hz: float):
    psd_dbmhz = p_dbm - 10.0 * np.log10(rbw_hz)
    header = "freq_hz,p_dbm,psd_dbm_per_hz\n"
    with open(path, "w", newline="") as f:
        f.write(header)
        for ff, pp, ss in zip(freq_hz, p_dbm, psd_dbmhz):
            f.write(f"{ff:.12g},{pp:.6f},{ss:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_freq", type=float, default=1e6)
    ap.add_argument("--max_freq", type=float, default=100e6)
    ap.add_argument("--points", type=int, default=1001)

    ap.add_argument("--rbw", type=float, nargs="+", default=[100e3, 10e3])
    ap.add_argument("--vbw", type=float, default=30e3)

    ap.add_argument("--dens", type=float, nargs="+", default=[3e-7, 6e-7, 1.2e-6])  # V/sqrt(Hz)

    ap.add_argument("--detector", type=str, default="RMS")  # "RMS" or "AVERage"
    ap.add_argument("--avg_on", type=int, default=1)        # 1/0
    ap.add_argument("--avg_count", type=int, default=20)

    ap.add_argument("--atten_db", type=float, default=10.0)
    ap.add_argument("--preamp_on", type=int, default=1)     # 1/0

    ap.add_argument("--generation_mode", type=int, default=1)
    ap.add_argument("--segment_ms", type=float, default=50.0)
    ap.add_argument("--amplitude_policy", type=str, default="strict")
    ap.add_argument("--noise_seed", type=int, default=12345)

    ap.add_argument("--plot", type=int, default=0)          # 1 enables plots per run
    args = ap.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = os.path.join(os.getcwd(), f"noise_test_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # Load defaults (instrument addresses, sample rate, etc.)
    defaults = utilities.load_defaults()

    # Compute sample rate + granularity like USER_main.py
    utilities.generation_mode_check(args.generation_mode, ["noise_white", "none", "none", "none"])
    srd = utilities.get_srd(args.generation_mode)
    dac_rate = defaults["waveform"]["sample_rate"]
    sample_rate = dac_rate / srd
    granularity = 256 // srd

    # Connect instruments once
    rm = pyvisa.ResourceManager()
    print("Connecting instruments...")

    awg = awg_control.connect_awg(
        rm,
        defaults["instruments"]["awg"]["address"],
        defaults["instruments"]["awg"]["timeout"],
    )

    sa = sa_control.connect_sa(
        rm,
        defaults["instruments"]["spectrum_analyzer"]["address"],
        defaults["instruments"]["spectrum_analyzer"]["timeout"],
    )

    # Fixed channel outputs: noise only on ch1
    channel_outputs = ["noise_white", "none", "none", "none"]

    # Configure SA once (some settings overwritten per-run below)
    # (We will re-call configure_sa per RBW condition to ensure RBW/VBW/avg are applied.)
    print("Connected.\n")

    # Results summary CSV
    summary_path = os.path.join(outdir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        f.write("rbw_hz,vbw_hz,noise_density_vrthz,median_psd_dbmhz,delta_db_vs_first\n")

    first_median = None

    try:
        for rbw_hz in args.rbw:
            for dens in args.dens:
                print("=" * 72)
                print(f"TEST | RBW={rbw_hz:g} Hz | VBW={args.vbw:g} Hz | dens={dens:g} V/rtHz")

                # Build waveform bundle(s) with requested noise density
                # We inject density + seed via globals that waveform_generation expects.
                # (waveform_generation reads these from USER_main globals? If not, it uses defaults in waveform_generation.)
                # Safer: set them in module namespace used by waveform_generation via built bundles metadata.
                #
                # Your waveform_generation currently respects calibrated density when 'noise_density_vrthz' is not None.
                # So we pass dens by temporarily setting waveform_generation.noise_density_vrthz if present.
                if hasattr(waveform_generation, "noise_density_vrthz"):
                    setattr(waveform_generation, "noise_density_vrthz", dens)
                if hasattr(waveform_generation, "noise_seed"):
                    setattr(waveform_generation, "noise_seed", args.noise_seed)
                # Set calibrated noise density in the place waveform_generation actually reads it from
                global noise_density_vrthz, noise_seed, noise_seed_current
                noise_density_vrthz = dens
                noise_seed = args.noise_seed
                noise_seed_current = None

                bundles = waveform_generation.build_waveform_bundles(
                    args.generation_mode,
                    [s.lower() for s in channel_outputs],
                    sample_rate,
                    granularity,
                    False,                 # preview
                    int(args.segment_ms),  # segment_length_ms
                )

                # Ensure metadata records the density (if your generator didn’t already)
                # (This is just for your sanity when you inspect bundles.)
                if bundles.get("ch1") is not None and hasattr(bundles["ch1"], "awg") and isinstance(bundles["ch1"].awg, dict):
                    bundles["ch1"].awg["noise_density_vrthz"] = dens
                    bundles["ch1"].awg["amplitude_policy"] = args.amplitude_policy

                # Configure and start AWG
                awg_control.configure_awg(awg, args.generation_mode, bundles)
                awg_control.start_output(awg, args.generation_mode)

                # Configure SA for this condition
                sa_control.configure_sa(
                    sa,
                    args.min_freq,
                    args.max_freq,
                    rbw_hz=rbw_hz,
                    vbw_hz=args.vbw,
                    points=args.points,
                    detector=args.detector,
                    avg_on=bool(args.avg_on),
                    avg_count=args.avg_count,
                    atten_db=args.atten_db,
                    preamp_on=bool(args.preamp_on),
                )

                # Sweep and read
                sa_control.initial_sweep(sa, sweeps=(args.avg_count if args.avg_on else 1))
                freq_hz, p_dbm, sa_settings = sa_control.get_spectrum_data(
                    sa,
                    plot=bool(args.plot),
                    plot_psd=bool(args.plot),
                )

                # Compute PSD and a median metric
                psd_dbmhz = p_dbm - 10.0 * np.log10(float(sa_settings.get("rbw_hz", rbw_hz)))
                med = median_dbmhz(freq_hz, psd_dbmhz, args.min_freq, args.max_freq)

                if first_median is None:
                    first_median = med
                delta = med - first_median if (np.isfinite(med) and np.isfinite(first_median)) else float("nan")

                print(
                    "SA VERIFY | "
                    f"ATTEN={sa_settings.get('atten_db')} dB | "
                    f"PREAMP={sa_settings.get('preamp_state')} | "
                    f"RBW={sa_settings.get('rbw_hz')} Hz | "
                    f"VBW={sa_settings.get('vbw_hz')} Hz | "
                    f"REF={sa_settings.get('ref_level_dbm')} dBm"
                )
                print(f"RESULT | median PSD = {med:.2f} dBm/Hz | delta vs first = {delta:.2f} dB")

                # Save full trace
                trace_name = f"trace_rbw{int(rbw_hz)}_dens{dens:.3e}.csv".replace("+", "")
                trace_path = os.path.join(outdir, trace_name)
                write_trace_csv(trace_path, freq_hz, p_dbm, rbw_hz=float(sa_settings.get("rbw_hz", rbw_hz)))

                # Append summary
                with open(summary_path, "a", newline="") as f:
                    f.write(f"{rbw_hz},{args.vbw},{dens},{med:.6f},{delta:.6f}\n")

        print("\nDone.")
        print(f"Summary: {summary_path}")
        print(f"Traces saved in: {outdir}")

        print("\nPASS/FAIL expectations:")
        print("- For fixed RBW, dens x2 should shift median PSD by ~ +6 dB; x4 by ~ +12 dB.")
        print("- For fixed dens, RBW 10k -> 100k: median PSD (dBm/Hz) should stay ~same, while raw dBm shifts ~+10 dB.")
    finally:
        try:
            sa_control.close_sa(sa)
        except Exception:
            pass


if __name__ == "__main__":
    main()
