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
# CONFIG LOADING / COMPATIBILITY BRIDGE
# ============================================================
_RUNTIME_CONFIG = None
noise_seed_current = None
sweep_current_freq = 0.0


def _channel_map(cfg, default):
    out = {}
    cfg = cfg or {}
    for ch in (1, 2, 3, 4):
        out[ch] = cfg.get(ch, cfg.get(str(ch), default))
    return out


def _apply_runtime_config(config):
    """
    Populate module globals from config.yaml so legacy modules that import
    USER_main continue to see the same names after the control surface move.
    """
    run_cfg = config.get("run", {}) or {}
    routing_cfg = run_cfg.get("routing", {}) or {}
    sweep_cfg = run_cfg.get("sweep_step", {}) or {}

    waveform_cfg = config.get("waveform", {}) or {}
    noise_cfg = waveform_cfg.get("noise", {}) or {}
    streaming_cfg = waveform_cfg.get("streaming", {}) or {}

    measurements_cfg = config.get("measurements", {}) or {}
    sa_cfg = config.get("spectrum_analyzer", {}) or {}
    osc_cfg = config.get("oscilloscope", {}) or {}

    current_noise_seed = globals().get("noise_seed_current", None)
    current_sweep_freq = globals().get("sweep_current_freq", None)

    compat_globals = {
        "generation_mode": int(run_cfg.get("generation_mode", 1)),
        "sweep_step_enable": bool(sweep_cfg.get("enable", False)),
        "sweep_step_start": float(sweep_cfg.get("start_hz", 0.0)),
        "sweep_step_stop": float(sweep_cfg.get("stop_hz", 0.0)),
        "sweep_step_count": int(sweep_cfg.get("count", 1)),
        "channel_1_output": str(routing_cfg.get("channel_1_output", "none")),
        "channel_2_output": str(routing_cfg.get("channel_2_output", "none")),
        "channel_3_output": str(routing_cfg.get("channel_3_output", "none")),
        "channel_4_output": str(routing_cfg.get("channel_4_output", "none")),
        "use_spectrum_analyzer": bool(measurements_cfg.get("use_spectrum_analyzer", True)),
        "use_oscilloscope": bool(measurements_cfg.get("use_oscilloscope", False)),
        "preview_waveform": bool(run_cfg.get("preview_waveform", False)),
        "save_data": bool(run_cfg.get("save_data", False)),
        "default_awg": dict(waveform_cfg.get("default_awg", {}) or {}),
        "component_gains": dict(waveform_cfg.get("component_gains", {}) or {}),
        "mix_headroom": float(waveform_cfg.get("mix_headroom", 1.0)),
        "amplitude_policy": str(waveform_cfg.get("amplitude_policy", "strict")),
        "clip_k_sigma": float(waveform_cfg.get("clip_k_sigma", 6.0)),
        "segment_length_ms": int(streaming_cfg.get("segment_length_ms", 10)),
        "continuous_noise_streaming": bool(streaming_cfg.get("continuous_noise_streaming", False)),
        "min_noise_segment_ms": int(streaming_cfg.get("min_noise_segment_ms", 50)),
        "noise_update_period_ms": int(streaming_cfg.get("noise_update_period_ms", 50)),
        "noise_fade_fraction": float(noise_cfg.get("fade_fraction", 0.0)),
        "noise_white_amplitude": float(noise_cfg.get("white_amplitude", 0.0)),
        "noise_pink_amplitude": float(noise_cfg.get("pink_amplitude", 0.0)),
        "noise_brown_amplitude": float(noise_cfg.get("brown_amplitude", 0.0)),
        "noise_density_vrthz": noise_cfg.get("density_vrthz"),
        "noise_bw_hz": noise_cfg.get("bandwidth_hz"),
        "noise_seed": noise_cfg.get("seed"),
        "noise_post_process_mode_ch": _channel_map(noise_cfg.get("post_process_mode_ch", {}), "none"),
        "noise_soft_clip_strength_ch": _channel_map(noise_cfg.get("soft_clip_strength_ch", {}), 2.0),
        "min_freq": float(sa_cfg.get("min_freq_hz", 0.0)),
        "max_freq": float(sa_cfg.get("max_freq_hz", 0.0)),
        "rbw_hz": float(sa_cfg.get("rbw_hz", 0.0)),
        "sa_sweep_points": int(sa_cfg.get("sweep_points", 1001)),
        "sa_vbw_hz": sa_cfg.get("vbw_hz"),
        "sa_detector": str(sa_cfg.get("detector", "RMS")),
        "sa_avg_on": bool(sa_cfg.get("avg_on", False)),
        "sa_avg_count": int(sa_cfg.get("avg_count", 1)),
        "sa_atten_db": float(sa_cfg.get("atten_db", 0.0)),
        "sa_preamp_on": bool(sa_cfg.get("preamp_on", False)),
        "sa_preamp_gain_db": float(sa_cfg.get("preamp_gain_db", 0.0)),
        "osc_time_scale": float(osc_cfg.get("time_scale", 1e-3)),
        "osc_time_position": float(osc_cfg.get("time_position", 0.0)),
        "osc_ch1_scale": float(osc_cfg.get("ch1_scale", 0.2)),
        "osc_ch2_scale": float(osc_cfg.get("ch2_scale", 0.2)),
        "osc_ch3_scale": float(osc_cfg.get("ch3_scale", 0.2)),
        "osc_ch4_scale": float(osc_cfg.get("ch4_scale", 0.2)),
        "osc_ch1_on": bool(osc_cfg.get("ch1_on", True)),
        "osc_ch2_on": bool(osc_cfg.get("ch2_on", False)),
        "osc_ch3_on": bool(osc_cfg.get("ch3_on", False)),
        "osc_ch4_on": bool(osc_cfg.get("ch4_on", False)),
        "osc_coupling_mode": str(osc_cfg.get("coupling_mode", "DC")),
        "osc_acquire_type": str(osc_cfg.get("acquire_type", "AVER")),
        "osc_acquire_count": int(osc_cfg.get("acquire_count", 16)),
        "osc_trigger_source": str(osc_cfg.get("trigger_source", "CHAN1")),
        "osc_trigger_level": float(osc_cfg.get("trigger_level", 0.0)),
        "osc_trigger_slope": str(osc_cfg.get("trigger_slope", "POS")),
        "osc_persistence": str(osc_cfg.get("persistence", "OFF")),
        "osc_grid": str(osc_cfg.get("grid", "FULL")),
    }

    globals().update(compat_globals)
    globals()["noise_seed_current"] = current_noise_seed
    globals()["sweep_current_freq"] = (
        float(current_sweep_freq)
        if current_sweep_freq is not None
        else float(compat_globals["sweep_step_start"])
    )


def load_runtime_config(force_reload=False):
    global _RUNTIME_CONFIG
    if force_reload or _RUNTIME_CONFIG is None:
        _RUNTIME_CONFIG = utilities.load_defaults()
        _apply_runtime_config(_RUNTIME_CONFIG)
    return _RUNTIME_CONFIG


# Load config at import time so dependent modules can safely read legacy globals.
load_runtime_config(force_reload=True)


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
        'frequencies': [1e6],   # Hz (1 Hz to ~2*sample_rate)
        'phases':      [0],
        'amplitudes':  [0.2],    # Vpk (-0.5 V to +0.5 V)
        'offsets':     [0],

        # awg analog scaling (instrument state)
        "awg": {
            "vpp": 1.0,
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
    # Reload config at runtime and refresh compatibility globals before any dependent logic runs.
    global segment_length_ms
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H;%M;%S")
    defaults = load_runtime_config(force_reload=True)
    globals()["noise_seed_current"] = None
    globals()["sweep_current_freq"] = sweep_step_start

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

    # ---------------------
    # Calibration/topology defaults
    # ---------------------
    cal_cfg = defaults.get("calibration", {}) or {}
    cal_enabled = bool(cal_cfg.get("enabled", False))
    output_topology = str(cal_cfg.get("output_topology", "single_leg")).strip().lower()
    output_leg = str(cal_cfg.get("output_leg", "DATA")).strip().upper()
    cw_cal_csv_path = cal_cfg.get("cw_cal_csv_path")
    comb_cfg = cal_cfg.get("combiner", {}) or {}
    combiner_enabled = bool(comb_cfg.get("enabled", False))
    il_csv_path = comb_cfg.get("il_csv_path")

    if output_topology not in ("single_leg", "differential_combined"):
        raise ValueError("calibration.output_topology must be 'single_leg' or 'differential_combined'.")
    if output_topology == "single_leg" and output_leg not in ("DATA", "DATA_BAR"):
        raise ValueError("calibration.output_leg must be DATA or DATA_BAR when output_topology=single_leg.")

    cw_table = None
    il_table = None
    if cal_enabled:
        sa_control.run_calibration_self_checks()
        if output_topology == "single_leg" and not cw_cal_csv_path:
            raise ValueError("Calibration enabled for single_leg but cw_cal_csv_path is missing.")
        if cw_cal_csv_path:
            cw_table = sa_control.CwCalTable(cw_cal_csv_path)
        if combiner_enabled:
            if not il_csv_path:
                raise ValueError("calibration.combiner.enabled=true but il_csv_path is missing.")
            il_table = sa_control.CombinerILTable(il_csv_path)

    topology_gain_db = (
        sa_control.TOPOLOGY_GAIN_DB_DIFFERENTIAL
        if (cal_enabled and output_topology == "differential_combined")
        else 0.0
    )
    print(
        "OUTPUT TOPOLOGY | "
        f"{output_topology.upper()} | leg={output_leg} | "
        f"cal_enabled={cal_enabled} | topology_gain_db={topology_gain_db:.4f} | "
        f"combiner_enabled={combiner_enabled} | cw_cal_csv={cw_cal_csv_path} | il_csv={il_csv_path}"
    )

    print("USING noise_density_vrthz =", noise_density_vrthz)


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
        sa_control.configure_sa(
            sa,
            min_freq,
            max_freq,
            rbw_hz=rbw_hz,
            vbw_hz=sa_vbw_hz,
            points=sa_sweep_points,
            detector=sa_detector,
            avg_on=sa_avg_on,
            avg_count=sa_avg_count,
            atten_db=sa_atten_db,
            preamp_on=sa_preamp_on,
        )


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

        # Waveform metadata snapshot (also used to derive commanded per-leg Vpk for calibration)
        active_key = None
        for k in ("ch1", "ch2", "ch3", "ch4"):
            if bundles.get(k) is not None:
                active_key = k
                break
        wb = bundles.get(active_key) if active_key else None
        wmeta = (wb.awg or {}) if wb is not None else {}

        amp_vpk_cmd = None
        if wmeta.get("amp_vpk_cmd", None) is not None:
            amp_vpk_cmd = float(wmeta.get("amp_vpk_cmd"))
        elif wmeta.get("vpp", None) is not None:
            amp_vpk_cmd = 0.5 * float(wmeta.get("vpp"))

        measurement_cal = None
        if cal_enabled:
            measurement_cal = sa_control.MeasurementCalibration(
                output_topology=output_topology,
                output_leg=output_leg,
                amp_vpk_cmd=amp_vpk_cmd,
                cw_table=cw_table,
                combiner_enabled=combiner_enabled,
                il_table=il_table,
                cw_cal_csv_path=cw_cal_csv_path,
                il_csv_path=il_csv_path,
            )

        freq_hz, p_dbm_raw, p_dbm, sa_settings = sa_control.get_spectrum_data(
            sa,
            plot=True,
            plot_psd=True,
            plot_both=True,
            preamp_gain_db=sa_preamp_gain_db,
            measurement_cal=measurement_cal,
            return_raw_and_corrected=True,
        )
        print(
            f"SA VERIFY | RBW={sa_settings.get('rbw_hz')} Hz | "
            f"VBW={sa_settings.get('vbw_hz')} Hz | "
            f"ATTEN={sa_settings.get('atten_db')} dB | "
            f"PREAMP={sa_settings.get('preamp_state')} | "
            f"REF={sa_settings.get('ref_level_dbm')} dBm | "
            f"TOPOLOGY={output_topology.upper()}({output_leg}) | "
            f"TOPOLOGY_GAIN={topology_gain_db:.4f} dB | "
            f"CAL_MEAN={sa_settings.get('cal_corr_db_mean', 0.0):.4f} dB"
        )
        
        rbw_hz = sa_settings.get("rbw_hz", np.nan)
        vbw_hz = sa_settings.get("vbw_hz", np.nan)

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
            "POWER_RAW (dBm)": float(pp_raw),
            "POWER (dBm)": float(pp),
            "CORR_DB (dB)": float(pp - pp_raw),
            "PREAMP.ON": int(bool(sa_settings.get("preamp_on", False))),
            "PREAMP.GAIN_DB": float(sa_settings.get("preamp_gain_db", 0.0) or 0.0),
            "POWER (dBm/Hz)": (float(pp) - 10.0 * np.log10(float(rbw_hz)) - float(sa_settings.get("preamp_gain_db", 0.0) or 0.0)) if (rbw_hz is not None and np.isfinite(rbw_hz) and float(rbw_hz) > 0) else np.nan,
            "STEP_FREQUENCY (Hz)": (float(step_freq) if step_freq is not None else np.nan),
            "RBW (Hz)": float(rbw_hz) if rbw_hz is not None else np.nan,
            "VBW (Hz)": float(vbw_hz) if vbw_hz is not None else np.nan,
            "OUTPUT_TOPOLOGY": output_topology.upper(),
            "OUTPUT_LEG": output_leg,
            "TOPOLOGY_GAIN_DB": float(topology_gain_db),
            "COMBINER_ENABLED": int(combiner_enabled),
            "COMBINER_IL_DB": (float(il_table.il_db(ff)) if (cal_enabled and combiner_enabled and il_table is not None) else np.nan),
            "CW_CAL_CSV": str(cw_cal_csv_path) if cw_cal_csv_path else "",
            "IL_CSV": str(il_csv_path) if il_csv_path else "",
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


        } for ff, pp_raw, pp in zip(freq_hz, p_dbm_raw, p_dbm)]
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
