# utilities.py
# Helper utilities for:
#   - Loading default config
#   - Validating user sine parameters
#   - Validating generation mode vs channel outputs
#   - Creating empty data files
#   - (placeholder) file writing

import datetime
import pandas as pd
import yaml
import os

import waveform_generation


def load_defaults():
    with open('config.yaml', 'r') as file:
        default_params = yaml.safe_load(file)
    return default_params


def validate_sine_param_lists(f, p, a, o):
    params = [f, p, a, o]
    n = len(params[0])
    for x in params:
        if len(x) == 0:
            raise Exception("Sine wave parameter lists cannot be empty.")
    if any(len(x) != n for x in params):
        raise Exception("All sine parameter lists must be equal in number of entries.")

    # TODO: If using spectrum analyzer, check that minimum frequency is 3 Hz
    #       and maximum frequency is 26.5 GHz


# --- utilities.py (fix get_srd) ---
def append_rows(file_path, rows):
    print("append_rows CALLED:", file_path, "nrows=", len(rows) if rows else 0)
    if not rows:
        print("append_rows: no rows to write")
        return
    df = pd.DataFrame(rows)
    df.to_csv(file_path, mode='a', index=False, header=False)
    print(f"append_rows: wrote {len(rows)} rows -> {file_path}")

def get_srd(mode):
    if mode in (1, 2, 3, 5, 6):
        return 1
    elif mode == 4:
        return 2
    else:
        raise Exception("Invalid generation mode given. Must be 1–6.")


def generation_mode_check(mode, channel_outputs):
    # Base valid waveform tokens
    base_valid_outputs = {
        'none',
        'sine1', 'sine2', 'sine3', 'sine4',
        'func1', 'func2', 'func3', 'func4',
        'noise_white', 'noise_pink', 'noise_brown'
    }

    # --- 1. Validate each channel output (supports superposition with "+") ---
    for i, output in enumerate(channel_outputs, 1):

        # Split on "+" to allow superposition (e.g., "sine1+noise_white")
        parts = output.split('+')

        for part in parts:
            if part not in base_valid_outputs:
                raise ValueError(
                    f"Invalid output '{part}' in '{output}' at position {i}. "
                    f"Each component must be one of {sorted(base_valid_outputs)}"
                )

        # Prevent illegal use of "none" in superposition
        if 'none' in parts and len(parts) > 1:
            raise ValueError(
                f"Invalid output '{output}' at position {i}. "
                f"'none' cannot be combined with other waveforms."
            )

    # --- 2. MODE VALIDATION (treat anything other than pure 'none' as active) ---

    def is_active(s):
        return s != 'none'

    if mode == 1:
        if not is_active(channel_outputs[0]):
            raise ValueError("Single channel mode: Channel 1 must have waveform.")

    elif mode == 2:
        if not is_active(channel_outputs[0]):
            raise ValueError("Single channel with markers mode: Channel 1 must have waveform.")

    elif mode == 3:
        if not is_active(channel_outputs[0]) or not is_active(channel_outputs[3]):
            raise ValueError("Dual channel mode: Channel 1 and Channel 4 must have waveform.")

    elif mode == 4:
        if any(not is_active(i) for i in channel_outputs):
            raise ValueError("Dual channel duplicate mode: All channels must specify the desired waveform.")

        if channel_outputs[0] != channel_outputs[2]:
            raise ValueError("Dual channel duplicate mode: Channel 1 and Channel 3 must have the same waveform.")

        if channel_outputs[1] != channel_outputs[3]:
            raise ValueError("Dual channel duplicate mode: Channel 2 and Channel 4 must have the same waveform.")

    elif mode == 5:
        if not is_active(channel_outputs[0]):
            raise ValueError("Dual channel with markers mode: Channel 1 must have waveform.")
        if not is_active(channel_outputs[1]):
            raise ValueError("Dual channel with markers mode: Channel 2 must have waveform.")

    elif mode == 6:
        if any(not is_active(i) for i in channel_outputs):
            raise ValueError("Four channel mode: All channels must specify the desired waveform.")

    else:
        raise Exception("Invalid generation mode given. Must be 1, 2, 3, 4, 5, or 6.")


def create_data_file(timestamp, mode, use_sa, use_osc):
    import os
    import pandas as pd

    filePath = r"C:\Users\ArnabLab\Documents\Measurement Instruments\Arbitrary Waveform Generator (Keysight M8195A)"
    os.makedirs(filePath, exist_ok=True)

    fileName = timestamp + "_AWG"
    if use_sa:
        fileName += "_SA"
    if use_osc:
        fileName += "_OSC"
    fileName += "_Data"

    fileLocation = os.path.join(filePath, f"{fileName}.dat")
    print("DATA FILE PATH:", fileLocation)

    cols = []

    # waveform columns (by mode)
    if mode in (1, 2):
        cols += ["W.Ch1 SAMPLE POINT", "W.Ch1 VOLTAGE (V)"]
    elif mode == 3:
        cols += ["W.Ch1 SAMPLE POINT", "W.Ch1 VOLTAGE (V)",
                 "W.Ch4 SAMPLE POINT", "W.Ch4 VOLTAGE (V)"]
    elif mode == 4:
        cols += ["W.Ch1,W.Ch3 SAMPLE POINT", "W.Ch1,W.Ch3 VOLTAGE (V)",
                 "W.Ch2,W.Ch4 SAMPLE POINT", "W.Ch2,W.Ch4 VOLTAGE (V)"]
    elif mode == 5:
        cols += ["W.Ch1 SAMPLE POINT", "W.Ch1 VOLTAGE (V)",
                 "W.Ch2 SAMPLE POINT", "W.Ch2 VOLTAGE (V)"]
    elif mode == 6:
        cols += ["W.Ch1 SAMPLE POINT", "W.Ch1 VOLTAGE (V)",
                 "W.Ch2 SAMPLE POINT", "W.Ch2 VOLTAGE (V)",
                 "W.Ch3 SAMPLE POINT", "W.Ch3 VOLTAGE (V)",
                 "W.Ch4 SAMPLE POINT", "W.Ch4 VOLTAGE (V)"]

    # SA columns
    if use_sa:
        cols += [
            "FREQUENCY (Hz)", "POWER (dBm)", "POWER (dBm/Hz)",
            "STEP_FREQUENCY (Hz)", "RBW (Hz)", "VBW (Hz)",
            # ---- waveform metadata (repeated per SA point) ----
            "WAVE.CH (str)",
            "AWG.VPP (Vpp)", "AWG.OFFS (V)", "AWG.LOAD (str)",
            "WAVE.FS (Sa/s)", "WAVE.N (samples)",
            "SINE.F_REQ (Hz)", "SINE.F_EFF (Hz)", "SINE.NCYCLES (str)",
            "NOISE.DENSITY (V/rtHz)",
            "AMP.POLICY (str)", "PEAK.UTIL (xFS)", "CLIP.PRED (bool)"
        ]

    # OSC columns
    if use_osc:
        cols += ["TIME (s)", "VOLTAGE (V)"]

    df = pd.DataFrame(columns=cols)
    df.to_csv(fileLocation, index=False)
    return fileLocation

def write_to_file(fileLocation):
    print("Work in progress! Nothing written to file.\n")

    # of = open(fileLocation, 'a')
    # with of as f:
    #     for p, v in zip(point, voltage):
    #         entry = '%s,%s' % (p, v)
    #         f.write(entry)
    #         f.write("\n")
    # of.close()

    return None

# def log_message():
#     pass
