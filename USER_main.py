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
channel_1_output = 'sine1+noise_white'
channel_2_output = 'none'
channel_3_output = 'none'
channel_4_output = 'none'

component_gains = {
    "sine1": 1.0,
    "noise_white": 1.5,
}
mix_headroom = 1.0

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
min_freq = 10.4999e9     # Hz
max_freq = 10.5001e9    # Hz
steps    = 0.00001e9      # Hz step used for RBW config


# 6) NOISE / STREAMING SETTINGS
# ------------------------------------------------------------
# segment_length_ms: length of each noise block (also used for mixed waveforms)
# continuous_noise_streaming: if True, NoiseStreamer will continuously
# regenerate noise segments on channels whose selector includes 'noise_*'.
segment_length_ms = 10    # Default: 10 ms per noise segment
continuous_noise_streaming = False

noise_fade_fraction = 0.0

noise_white_amplitude = 0.5 # 0.0 to 1.0 recommended (scaling of noise)
noise_pink_amplitude = 0.5
noise_brown_amplitude = 0.5

# Calibrated white-noise density in V/sqrt(Hz). If None, disabled.
noise_density_vrthz = None

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
    print("sine1 params freq =", sweep_current_freq)

    return {
        # signal definition (software, Volts)
        'frequencies': [10.5e6],   # Hz (1 Hz to ~2*sample_rate)
        'phases':      [0],
        'amplitudes':  [0.5],    # Vpk (-0.5 V to +0.5 V)
        'offsets':     [0],

        # awg analog scaling (instrument state)
        "awg": {
            "vpp": 1.0,
            "offs": 0.0,
            "load": "INF",      # "50" or "INF" (Hi-Z)
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

    utilities.generation_mode_check(generation_mode, channel_outputs)
    srd = utilities.get_srd(generation_mode)

    dac_rate = defaults['waveform']['sample_rate']
    sample_rate = dac_rate / srd
    granularity = 256 // srd

    global noise_density_vrthz
    noise_density_vrthz = defaults.get("waveform", {}).get("noise_density_vrthz", noise_density_vrthz)

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
        sa_control.configure_sa(sa, min_freq, max_freq, steps)

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
    def log_sa(step_freq=None):
        if not use_spectrum_analyzer or sa is None:
            return

        print("LOG_SA writing step =", step_freq)
        sa_control.initial_sweep(sa)

        # requires sa_control.get_spectrum_data(sa, plot=False) -> (freq_hz, p_dbm)
        freq_hz, p_dbm = sa_control.get_spectrum_data(sa, plot=True)

        rows = [{
            "FREQUENCY (Hz)": float(ff),
            "POWER (dBm)": float(pp),
            "STEP_FREQUENCY (Hz)": (float(step_freq) if step_freq is not None else np.nan),
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
        )
        noise_streamer.start()

    # ---------------------
    # Sweep vs non-sweep
    # ---------------------
    if sweep_step_enable:
        freqs = np.linspace(sweep_step_start, sweep_step_stop, sweep_step_count)

        global sweep_current_freq
        for f in freqs:
            sweep_current_freq = float(f)

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
            log_sa(step_freq=sweep_current_freq)

    else:
        # Start AWG once
        awg_control.start_output(awg, generation_mode)

        # Save SA data once
        log_sa(step_freq=None)

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
