import datetime
import numpy as np
import pyvisa

import waveform_generation
import awg_control
import sa_control
import osc_control
import utilities
import noise_streaming


# ============================================================
# USER CONFIGURATION SECTION
# ============================================================

# 1) AWG GENERATION MODE & ROUTING
# ------------------------------------------------------------
# Possible modes:
#   1: Single channel (Ch.1)
#   2: Single channel with markers (Ch.1)
#   3: Dual channel (Ch.1 and Ch.4)
#   4: Dual channel duplicate (Ch.1 = Ch.3, Ch.2 = Ch.4)
#   5: Dual channel with markers (Ch.1 and Ch.2)
#   6: Four channel (Ch.1, Ch.2, Ch.3, and Ch.4)
generation_mode = 1

sweep_step_enable = False
sweep_step_start = 5e9
sweep_step_stop  = 12e9
sweep_step_count = 15
sweep_current_freq = sweep_step_start



# 2) WAVEFORM SELECTION PER CHANNEL
# ------------------------------------------------------------
# These strings control what appears on each channel.
# Options include:
#   - 'none'
#   - 'sine1', 'sine2', 'sine3', 'sine4'
#   - 'func1', 'func2', 'func3', 'func4'
#   - 'noise_white', 'noise_pink', 'noise_brown'
#   - or any '+' combination, e.g. 'sine1+noise_white'
#
# NOTE: Channel-to-DAC mapping depends on generation_mode (see above).
channel_1_output = 'noise_white'
channel_2_output = 'none'
channel_3_output = 'none'
channel_4_output = 'none'

# Default AWG analog scaling used when a waveform selector does NOT define its own "awg" dict
# (e.g., noise_white/noise_pink/noise_brown).
default_awg = {
    "vpp": 1.0,     # <-- change this to whatever you want (e.g. 2.0, 3.0, etc.)
    "offs": 0.0,
    "load": "50",   # "50" or "INF"
}

component_gains = {
    "sine1": 1.0,
    "noise_white": 1.0,
}
mix_headroom = 1.0
# Amplitude policy:
#   - 'strict': no silent rescaling; abort if the mix would clip the requested Vpp.
#   - 'safe'  : apply one global scale factor to fit within Vpp (logs applied_scale_factor).
amplitude_policy = 'safe'

# Conservative sigma multiplier used by optional clip predictor modes (kept for future use).
clip_k_sigma = 6.0


# 3) INSTRUMENT USAGE FLAGS
# ------------------------------------------------------------
# AWG is always used. Enable/disable measurement instruments here.
use_spectrum_analyzer = True
use_oscilloscope = True


# 4) PREVIEW & DATA LOGGING
# ------------------------------------------------------------
# preview_waveform: show matplotlib preview of generated waveform(s)
# save_data: create an empty .dat file with appropriate columns.
preview_waveform = True
save_data = False


# 5) SPECTRUM ANALYZER SWEEP SETTINGS
# ------------------------------------------------------------
# Only used if use_spectrum_analyzer = True.
min_freq = 1e6     # Hz
max_freq = 100e6    # Hz
rbw_hz    = 100e3      # SA resolution bandwidth (RBW) in Hz
sa_sweep_points = 1001      # Number of points per SA sweep
sa_vbw_hz = 30e3

sa_detector = "RMS"      # "RMS" or "AVERage" depending on SA support
sa_avg_on   = True       # enable SA trace averaging
sa_avg_count = 20        # number of sweeps to average
sa_atten_db = 10
sa_preamp_on = True

# 6) NOISE / STREAMING SETTINGS
# ------------------------------------------------------------
# segment_length_ms: length of each noise block (also used for mixed waveforms)
# continuous_noise_streaming: if True, NoiseStreamer will continuously
# regenerate noise segments on channels whose selector includes 'noise_*'.
segment_length_ms = 50    # Default: 10 ms per noise segment
continuous_noise_streaming = False
min_noise_segment_ms = 50          # enforce if any selector contains noise_
noise_update_period_ms = 50        # streamer update cadence (ms)


noise_fade_fraction = 0.0

noise_white_amplitude = 0.01 # 0.0 to 1.0 recommended (scaling of noise)
noise_pink_amplitude = 0.5
noise_brown_amplitude = 0.5

# Calibrated white-noise density in V/sqrt(Hz). If None, disabled.
noise_density_vrthz = 3e-7
noise_seed = 12345   # set None to disable deterministic noise

# Noise Post Processing:
noise_post_process_mode_ch = {
    1: "none",
    2: "none",
    3: "none",
    4: "none",
}

noise_soft_clip_strength_ch = {
    1: 0.0,
    2: 2.0,
    3: 3.0,
    4: 2.0,
}



# 7) OSCILLOSCOPE SETTINGS
# ------------------------------------------------------------

# Horizontal (time base)
osc_time_scale = 1e-3       # seconds/div (default: 1 ms/div)
osc_time_position = 0       # horizontal position (0 = center)

# Vertical scale per channel
osc_ch1_scale = 0.2         # volts/div
osc_ch2_scale = 0.2         # volts/div
osc_ch3_scale = 0.2
osc_ch4_scale = 0.2

# Channel enable/disable
osc_ch1_on = True
osc_ch2_on = False
osc_ch3_on = False
osc_ch4_on = False

# Coupling mode options: AC, DC, GND
osc_coupling_mode = "DC"    # applied to all enabled channels

# Acquisition mode
osc_acquire_type = "AVER"   # NORM, AVER, HRES
osc_acquire_count = 16      # averaging count if AVER mode

# Trigger settings
osc_trigger_source = "CHAN1"
osc_trigger_level = 0.0     # volts
osc_trigger_slope = "POS"   # POS or NEG

# Display settings
osc_persistence = "OFF"     # OFF, SHORT, MEDIUM, LONG, INF
osc_grid = "FULL"           # FULL, HALF, NONE


# ============================================================
# USER-DEFINED WAVEFORM PARAMETERS
# ============================================================
# STEP A: USER DEFINES WAVEFORM PARAMETER FUNCTIONS
#
# Option 1: Sine combination waveforms.
#   Each function returns dictionaries with:
#     'frequencies', 'phases', 'amplitudes', 'offsets'
#   These are used when channel_X_output contains 'sineN'.
# ------------------------------------------------------------

def sine_combination_waveform1():
    global sweep_current_freq

    return {
        # signal definition (software, Volts)
        'frequencies': [10.5e6],   # Hz (1 Hz to ~2*sample_rate)
        'phases':      [0],
        'amplitudes':  [0.4],    # Vpk (-0.5 V to +0.5 V)
        'offsets':     [0],

        # awg analog scaling (instrument state)
        "awg": {
            "vpp": 3.0,
            "offs": 0.0,
            "load": "50",      # "50" or "INF" (Hi-Z)
        }
    }
    


def sine_combination_waveform2():
    global sweep_current_freq
    return {
        # signal definition (software, Volts)
        'frequencies': [],   # Hz (1 Hz to ~2*sample_rate)
        'phases':      [],
        'amplitudes':  [],    # Vpk (-0.5 V to +0.5 V)
        'offsets':     [],

        # awg analog scaling (instrument state)
        "awg": {
            "vpp": 1.0,
            "offs": 0.0,
            "load": "INF",      # "50" or "INF" (Hi-Z)
        }
    }


def sine_combination_waveform3():
    global sweep_current_freq
    return {
        # signal definition (software, Volts)
        'frequencies': [],   # Hz (1 Hz to ~2*sample_rate)
        'phases':      [],
        'amplitudes':  [],    # Vpk (-0.5 V to +0.5 V)
        'offsets':     [],

        # awg analog scaling (instrument state)
        "awg": {
            "vpp": 1.0,
            "offs": 0.0,
            "load": "INF",      # "50" or "INF" (Hi-Z)
        }
    }


def sine_combination_waveform4():
    global sweep_current_freq
    return {
        # signal definition (software, Volts)
        'frequencies': [],   # Hz (1 Hz to ~2*sample_rate)
        'phases':      [],
        'amplitudes':  [],    # Vpk (-0.5 V to +0.5 V)
        'offsets':     [],

        # awg analog scaling (instrument state)
        "awg": {
            "vpp": 1.0,
            "offs": 0.0,
            "load": "INF",      # "50" or "INF" (Hi-Z)
        }
    }


# Option 2: User-defined custom waveforms.
#   These are placeholders; you can implement your own logic.
#   Used when channel_X_output contains 'funcN'.
# ------------------------------------------------------------

def user_waveform1():
    Fs = 1e9 #effective sample rate (1 GSa/s)
    duration = 2e-6 # 2 microseconds
    N = int(Fs * duration) # number of samples

    t = np.linspace(0, 1, N, endpoint=False)

    # Sawtooth ramps from -1 to +1 over duration
    y = 2 * t - 1

    return y


def user_waveform2():
    return 0


def user_waveform3():
    return 0


def user_waveform4():
    return 0


# ============================================================
# MAIN PROGRAM EXECUTION (DO NOT MODIFY)
# ============================================================

def main():
    # Load defaults and user input
    global segment_length_ms
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H;%M;%S")
    defaults = utilities.load_defaults()

    # ALWAYS create a data file (per your request)
    file_path = utilities.create_data_file(
        timestamp,
        generation_mode,
        use_spectrum_analyzer,
        use_oscilloscope,
    )

    channel_outputs = [
        channel_1_output.lower(),
        channel_2_output.lower(),
        channel_3_output.lower(),
        channel_4_output.lower(),
    ]
    has_noise = any("noise_" in s for s in channel_outputs)
    if has_noise:
        segment_length_ms = max(segment_length_ms, min_noise_segment_ms)


    utilities.generation_mode_check(generation_mode, channel_outputs)
    srd = utilities.get_srd(generation_mode)

    dac_rate = defaults['waveform']['sample_rate']
    sample_rate = dac_rate / srd
    granularity = 256 // srd

    global noise_density_vrthz
    noise_density_vrthz = defaults.get("waveform", {}).get("noise_density_vrthz", noise_density_vrthz)
    global noise_seed
    noise_seed = defaults.get("waveform", {}).get("noise_seed", noise_seed)
    
    global sa_detector, sa_avg_on, sa_avg_count
    sa_detector = defaults.get("spectrum_analyzer", {}).get("detector", sa_detector)
    sa_avg_on = defaults.get("spectrum_analyzer", {}).get("avg_on", sa_avg_on)
    sa_avg_count = defaults.get("spectrum_analyzer", {}).get("avg_count", sa_avg_count)

    global sa_atten_db, sa_preamp_on, sa_vbw_hz
    sa_atten_db = defaults.get("spectrum_analyzer", {}).get("atten_db", sa_atten_db)
    sa_preamp_on = defaults.get("spectrum_analyzer", {}).get("preamp_on", sa_preamp_on)
    sa_vbw_hz = defaults.get("spectrum_analyzer", {}).get("video_bandwidth", sa_preamp_on)

    # Build initial waveform bundle(s)
    bundles = waveform_generation.build_waveform_bundles(
        generation_mode,
        channel_outputs,
        sample_rate,
        granularity,
        preview_waveform,
        segment_length_ms,
    )

    # Connect to, configure, and upload waveform(s) to instruments
    rm = pyvisa.ResourceManager()
    print("Instruments Connected:\n")

    awg = awg_control.connect_awg(
        rm,
        defaults['instruments']['awg']['address'],
        defaults['instruments']['awg']['timeout'],
    )
    awg_control.configure_awg(awg, generation_mode, bundles)

    sa = None
    if use_spectrum_analyzer:
        sa = sa_control.connect_sa(
            rm,
            defaults['instruments']['spectrum_analyzer']['address'],
            defaults['instruments']['spectrum_analyzer']['timeout'],
        )
        sa_control.configure_sa(sa, min_freq, max_freq, rbw_hz=rbw_hz, vbw_hz=sa_vbw_hz, points=sa_sweep_points, detector=sa_detector, avg_on=sa_avg_on, avg_count=sa_avg_count, atten_db=sa_atten_db, preamp_on=sa_preamp_on)


    osc = None
    if use_oscilloscope:
        osc = osc_control.connect_osc(
            rm,
            defaults['instruments']['oscilloscope']['address'],
            defaults['instruments']['oscilloscope']['timeout'],
        )
        osc_settings = {
            'time_scale':     osc_time_scale,
            'time_position':  osc_time_position,
            'ch1_on':         osc_ch1_on,
            'ch2_on':         osc_ch2_on,
            'ch3_on':         osc_ch3_on,
            'ch4_on':         osc_ch4_on,
            'ch1_scale':      osc_ch1_scale,
            'ch2_scale':      osc_ch2_scale,
            'ch3_scale':      osc_ch3_scale,
            'ch4_scale':      osc_ch4_scale,
            'coupling':       osc_coupling_mode,
            'acquire_type':   osc_acquire_type,
            'acquire_count':  osc_acquire_count,
            'trig_source':    osc_trigger_source,
            'trig_level':     osc_trigger_level,
            'trig_slope':     osc_trigger_slope,
            'persistence':    osc_persistence,
            'grid':           osc_grid,
        }
        osc_control.configure_osc(osc, osc_settings)

    # ---------------------
    # SA logging helper
    # ---------------------
    def log_sa(bundles, step_freq=None):
        if not use_spectrum_analyzer or sa is None:
            return

        print("LOG_SA writing step =", step_freq)
        sa_control.initial_sweep(sa, sweeps=(sa_avg_count if sa_avg_on else 1))

        # requires sa_control.get_spectrum_data(sa, plot=False) -> (freq_hz, p_dbm)
        preamp_gain_db = float(cfg.get("spectrum_analyzer", {}).get("preamp_gain_db", 0.0))
        freq_hz, p_dbm, sa_settings = sa_control.get_spectrum_data(sa, plot=True, plot_psd=True, preamp_gain_db=preamp_gain_db)

        print(
            f"SA VERIFY | RBW={sa_settings.get('rbw_hz')} Hz | "
            f"VBW={sa_settings.get('vbw_hz')} Hz | "
            f"ATTEN={sa_settings.get('atten_db')} dB | "
            f"PREAMP={sa_settings.get('preamp_state')} | "
            f"REF={sa_settings.get('ref_level_dbm')} dBm"
        )
        
        rbw_hz = sa_settings.get("rbw_hz", np.nan)
        vbw_hz = sa_settings.get("vbw_hz", np.nan)

        # -----------------------------
        # Waveform metadata snapshot
        # -----------------------------
        active_key = None
        for k in ("ch1", "ch2", "ch3", "ch4"):
            if bundles.get(k) is not None:
                active_key = k
                break

        wb = bundles.get(active_key) if active_key else None
        wmeta = (wb.awg or {}) if wb is not None else {}

        wave_ch = active_key if active_key else ""
        awg_vpp = float(wmeta.get("vpp", np.nan)) if wmeta.get("vpp", None) is not None else np.nan
        awg_offs = float(wmeta.get("offs", np.nan)) if wmeta.get("offs", None) is not None else np.nan
        awg_load = str(wmeta.get("load", ""))
        
        wave_fs = float(wmeta.get("Fs", np.nan)) if wmeta.get("Fs", None) is not None else np.nan
        wave_n  = int(wmeta.get("N", -1)) if wmeta.get("N", None) is not None else -1
        tseg_s = (float(wave_n) / float(wave_fs)) if (np.isfinite(wave_fs) and wave_fs > 0 and wave_n > 0) else np.nan
        update_s = (float(noise_update_period_ms) / 1000.0) if noise_update_period_ms is not None else np.nan

        sine_f_req = float(wmeta.get("f_req", np.nan)) if wmeta.get("f_req", None) is not None else np.nan
        sine_f_eff = float(wmeta.get("f_eff", np.nan)) if wmeta.get("f_eff", None) is not None else np.nan

        sine_ncycles = str(wmeta.get("Ncycles", ""))

        noise_density = wmeta.get("noise_density_vrthz", None)
        noise_density = float(noise_density) if noise_density is not None else np.nan
        print(f"NOISE VERIFY | NOISE.DENSITY (V/rtHz) = {noise_density}")

        amp_policy = str(wmeta.get("amplitude_policy", ""))
        peak_util  = float(wmeta.get("peak_utilization", np.nan)) if wmeta.get("peak_utilization", None) is not None else np.nan
        clip_pred  = bool(wmeta.get("clip_predicted", False)) if wmeta.get("clip_predicted", None) is not None else False


        rows = [{
            "FREQUENCY (Hz)": float(ff),
            "POWER (dBm)": float(pp),
            "PREAMP.ON": int(bool(sa_settings.get("preamp_on", False))),
            "PREAMP.GAIN_DB": float(sa_settings.get("preamp_gain_db", 0.0) or 0.0),
            "POWER (dBm/Hz)": (float(pp) - 10.0 * np.log10(float(rbw_hz)) - float(sa_settings.get("preamp_gain_db", 0.0) or 0.0)) if (rbw_hz is not None and np.isfinite(rbw_hz) and float(rbw_hz) > 0) else np.nan,
            "STEP_FREQUENCY (Hz)": (float(step_freq) if step_freq is not None else np.nan),
            "RBW (Hz)": float(rbw_hz) if rbw_hz is not None else np.nan,
            "VBW (Hz)": float(vbw_hz) if vbw_hz is not None else np.nan,
            "TSEG (s)": tseg_s,
            "NOISE.UPDATE (s)": update_s,            
            "WAVE.CH (str)": wave_ch,
            "AWG.VPP (Vpp)": awg_vpp,
            "AWG.OFFS (V)": awg_offs,
            "AWG.LOAD (str)": awg_load,
            "WAVE.FS (Sa/s)": wave_fs,
            "WAVE.N (samples)": wave_n,
            "SINE.F_REQ (Hz)": sine_f_req,
            "SINE.F_EFF (Hz)": sine_f_eff,
            "SINE.NCYCLES (str)": sine_ncycles,
            "NOISE.DENSITY (V/rtHz)": noise_density,
            "AMP.POLICY (str)": amp_policy,
            "PEAK.UTIL (xFS)": peak_util,
            "CLIP.PRED (bool)": clip_pred,


        } for ff, pp in zip(freq_hz, p_dbm)]
        utilities.append_rows(file_path, rows)

        

    # Optional continuous noise + static waveform mixing
    noise_streamer = None
    if continuous_noise_streaming:
        noise_streamer = noise_streaming.NoiseStreamer(
            awg=awg,
            mode=generation_mode,
            channel_outputs=channel_outputs,
            Fs=sample_rate,
            granularity=granularity,
            segment_length_ms=segment_length_ms,
            noise_update_period_ms=noise_update_period_ms,

        )
        noise_streamer.start()

    # ---------------------
    # Sweep vs non-sweep
    # ---------------------
    if sweep_step_enable:
        freqs = np.linspace(sweep_step_start, sweep_step_stop, sweep_step_count)

        global sweep_current_freq
        for step_idx, f in enumerate(freqs):
            sweep_current_freq = float(f)
            if noise_seed is not None:
                noise_seed_current = int(noise_seed) + int(step_idx)
            else:
                noise_seed_current = None
            globals()["noise_seed_current"] = noise_seed_current

            bundles = waveform_generation.build_waveform_bundles(
                generation_mode,
                channel_outputs,
                sample_rate,
                granularity,
                preview_waveform,
                segment_length_ms
            )
            awg_control.configure_awg(awg, generation_mode, bundles)
            awg_control.start_output(awg, generation_mode)

            # save SA data for every frequency step
            log_sa(bundles, step_freq=sweep_current_freq)

    else:
        # Start AWG once
        awg_control.start_output(awg, generation_mode)

        # Save SA data once
        log_sa(bundles, step_freq=None)

    # Stop streaming (if used)
    if noise_streamer is not None:
        noise_streamer.stop()

    # Close instruments
    # awg_control.close_awg(awg)
    if use_spectrum_analyzer and sa is not None:
        sa_control.close_sa(sa)
    if use_oscilloscope and osc is not None:
        osc_control.close_osc(osc)

if __name__ == "__main__":
    main()
