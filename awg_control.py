# awg_control.py
# Low-level control for Keysight M8195A AWG:
#   - Connect
#   - Configure DAC mode and memory
#   - Upload waveform segments
#   - Optional per-channel voltage settings
#   - Helpers for streaming / double-buffer updates

import pyvisa
import numpy as np


# ============================================================
# CONNECTION
# ============================================================

def _csv(arr: np.ndarray) -> str:
    """Legacy helper: convert numpy array to comma-separated string."""
    return ','.join(arr.astype(str))


def _try_write(awg, cmd: str) -> bool:
    """Write SCPI command; return True if no instrument error."""
    awg.write(cmd)
    err = awg.query(":SYST:ERR?").strip()
    return err.startswith("+0")

def _ensure_symmetric_mode(awg, ch: int) -> None:
    """
    Force symmetric mode around 0 V if the AWG supports it.
    On M8195A this typically means VTERM=0 and OFFS=0.
    """
    # Some firmwares use :VOLT{ch}:TERM, others may not support it.
    _try_write(awg, f":VOLT{ch}:TERM 0")
    _try_write(awg, f":VOLT{ch}:OFFS 0")

def connect_awg(rm, address, timeout):
    """Open a VISA connection to the AWG and print identity."""
    awg = rm.open_resource(address)
    awg.timeout = timeout
    idn = awg.query("*IDN?")
    print("AWG Identity:", idn.strip())
    return awg


# ============================================================
# MODE / MEMORY CONFIGURATION
# ============================================================

def operation_mode(awg, mode: int):
    """
    Configure DAC mode and memory divider based on desired channel / marker mode.

    mode:
      1: Single channel (Ch.1)
      2: Single channel with markers (Ch.1)
      3: Dual channel (Ch.1 and Ch.4)
      4: Dual channel duplicate (Ch.1 = Ch.3, Ch.2 = Ch.4)
      5: Dual channel with markers (Ch.1 and Ch.2)
      6: Four channel (Ch.1, Ch.2, Ch.3, Ch.4)
    """
    if mode == 1:
        awg.write(':INSTrument:DACMode SINGle')
        awg.write(':INSTrument:MEMory:EXTended:RDIVider DIV1')
    elif mode == 2:
        awg.write(':INSTrument:DACMode MARKer')
        awg.write(':INSTrument:MEMory:EXTended:RDIVider DIV1')
    elif mode == 3:
        awg.write(':INSTrument:DACMode DUAL')
        awg.write(':INSTrument:MEMory:EXTended:RDIVider DIV1')
    elif mode == 4:
        awg.write(':INSTrument:DACMode DCDuplicate')
        awg.write(':INSTrument:MEMory:EXTended:RDIVider DIV2')
    elif mode == 5:
        awg.write(':INSTrument:DACMode DCMarker')
        awg.write(':INSTrument:MEMory:EXTended:RDIVider DIV1')
    elif mode == 6:
        awg.write(':INSTrument:DACMode FOUR')
        awg.write(':INSTrument:MEMory:EXTended:RDIVider DIV1')
    else:
        raise ValueError(f"Invalid generation mode number: {mode}")


# ============================================================
# WAVEFORM UPLOAD HELPERS
# ============================================================

def _apply_awg_settings(awg, ch_num: int, awg_cfg: dict | None):
    """
    Apply per-channel output settings.

    Behavior:
    - If user does NOT provide an "offs" key at all -> force symmetric (TERM=0, OFFS=0)
    - If user provides "offs" (including 0.0) -> use it as requested (do not auto-force TERM)
    - Always writes OFFS explicitly (fixes previous bug)
    """
    if not awg_cfg:
        return

    # Pull values
    vpp  = awg_cfg.get("vpp", None)
    load = awg_cfg.get("load", None)

    # Detect whether user explicitly set an offset key
    user_set_offs_key = ("offs" in awg_cfg)
    offs_val = awg_cfg.get("offs", None)

    # Turn output on first
    awg.write(f":OUTP{ch_num} ON")

    # If user did NOT specify offs at all, enforce symmetric around 0V.
    # (This is the "auto symmetric mode" behavior you wanted.)
    if not user_set_offs_key:
        # Best-effort: these may error on some firmwares; we don't want to crash
        _try_write(awg, f":VOLT{ch_num}:TERM 0")
        _try_write(awg, f":VOLT{ch_num}:OFFS 0")
        offs_to_set = 0.0
    else:
        # User explicitly requested a DC offset (even 0.0)
        offs_to_set = 0.0 if offs_val is None else float(offs_val)

    # Apply amplitude (Vpp) if provided
    if vpp is not None:
        awg.write(f":VOLT{ch_num} {float(vpp)}")

    # Always apply OFFS explicitly (this was missing/buggy before)
    awg.write(f":VOLT{ch_num}:OFFS {offs_to_set}")

    # Apply load if provided (firmware-dependent; try both)
    if load is not None:
        for cmd in (f":OUTP{ch_num}:LOAD {load}", f":OUTP{ch_num}:IMP {load}"):
            awg.write(cmd)
            if awg.query(":SYST:ERR?").strip().startswith("+0"):
                break

    # Debug / readback (keep your existing style)
    print("SYST:ERR?", awg.query("SYST:ERR?").strip())
    print("VOLT?", awg.query(f":VOLT{ch_num}?").strip())
    print("OFFS?", awg.query(f":VOLT{ch_num}:OFFS?").strip())
    # TERM? may not exist on all firmwares; keep it best-effort
    try:
        print("TERM?", awg.query(f":VOLT{ch_num}:TERM?").strip())
    except Exception:
        pass
    print("ERR?",  awg.query(":SYST:ERR?").strip())

def _upload_legacy_list(awg, mode: int, waveforms):
    """
    Legacy upload path for simple 1-channel use (mode 1 only).

    waveforms: list-like of np.int8 arrays, first element goes to Ch1, segment 1.
    """
    if mode != 1:
        raise Exception("Legacy list upload supports mode 1 only.")

    w1 = np.asarray(waveforms[0], dtype=np.int8)
    n = w1.size

    # Clear all segments
    awg.write(':TRACe:DELete:ALL')

    # Define segment 1 on Channel 1
    awg.write(f':TRACe1:DEFine 1,{n}')

    # Binary upload of waveform into segment 1, starting at offset 0
    awg.write_binary_values(
        ':TRACe1:DATA 1,0,',
        w1,
        datatype='b',
        is_big_endian=True
    )

    print(f"Uploaded legacy waveform to Channel 1, segment 1, {n} samples.")


def _upload_bundles(awg, mode: int, bundles: dict):
    """
    Upload WaveBundle-like objects to the AWG.

    bundles: dict with keys 'ch1'..'ch4' mapping to objects with:
      .samples -> np.int8 (or castable) array
      .markers -> optional, ignored for front-panel use

    Segment numbers are assigned sequentially (1, 2, 3, ...) in the
    order channels are actually loaded.
    """
    # Clear all segments
    awg.write(':TRACe:DELete:ALL')
    seg = 1  # segment counter, shared across channels

    def load(ch_key: str, ch_num: int):
        nonlocal seg
        b = bundles.get(ch_key)
        if b is None:
            return

        # Ensure int8 array
        data = np.asarray(b.samples, dtype=np.int8)
        N = data.size
        if N <= 0:
            print(f"Warning: bundle '{ch_key}' has zero samples; skipping.")
            return

        # Define a new segment of length N on the given channel
        awg.write(f':TRACe{ch_num}:DEFine {seg},{N}')

        # Upload entire segment in a single binary transfer starting at offset 0
        awg.write_binary_values(
            f':TRACe{ch_num}:DATA {seg},0,',
            data,
            datatype='b',
            is_big_endian=True
        )


        _apply_awg_settings(awg, ch_num, getattr(b, "awg", None))


        print(f"Uploaded bundle '{ch_key}' to Channel {ch_num}, segment {seg}, {N} samples.")
        seg += 1

    # NOTE: DACMode is already set by operation_mode(); here we just map bundles → channels.
    if mode == 1:
        load('ch1', 1)
    elif mode == 2:
        # Single channel with markers (Ch.1)
        load('ch1', 1)
    elif mode == 3:
        # Dual channel (Ch.1 and Ch.4)
        load('ch1', 1)
        load('ch4', 4)
    elif mode == 4:
        # Dual channel duplicate (Ch.1 = Ch.3, Ch.2 = Ch.4)
        load('ch1', 1)
        load('ch3', 3)
        load('ch2', 2)
        load('ch4', 4)
    elif mode == 5:
        # Dual channel with markers (Ch.1 and Ch.2)
        load('ch1', 1)
        load('ch2', 2)
    elif mode == 6:
        # Four channel (Ch.1, Ch.2, Ch.3, Ch.4)
        load('ch1', 1)
        load('ch2', 2)
        load('ch3', 3)
        load('ch4', 4)
    else:
        raise ValueError(f"Invalid generation mode: {mode}")


# ============================================================
# HIGH-LEVEL CONFIGURE / RUN
# ============================================================

def configure_awg(awg, mode: int, payload, v1=None, v2=None, v3=None, v4=None):
    """
    Configure AWG for waveform generation and upload data.

    payload can be:
      - list[np.int8] (legacy 1-ch waveform for mode 1)
      - dict of bundles from waveform_generation.build_waveform_bundles
    """
    # Stop any ongoing generation
    awg.write(':ABORt')

    # Set up DAC mode and memory divider
    operation_mode(awg, mode)

    # Arm trigger behavior
    awg.write(':ARM:TRIGger:OPERation SYNChronous')

    # Upload waveforms
    if isinstance(payload, dict):
        _upload_bundles(awg, mode, payload)
    else:
        _upload_legacy_list(awg, mode, payload)



# --- Helpers for streaming / double-buffer updates ---

def update_segment_samples(awg, ch_num: int, seg_num: int, samples: np.ndarray):
    """
    Overwrite a specific segment on a specific channel with new samples.
    Assumes segment exists or will be redefined to N.
    """
    data = np.asarray(samples, dtype=np.int8)
    N = data.size
    awg.write(f":TRACe{ch_num}:DEFine {seg_num},{N}")
    awg.write_binary_values(
        f":TRACe{ch_num}:DATA {seg_num},0,",
        data,
        datatype='b',
        is_big_endian=True
    )


def select_segment(awg, ch_num: int, seg_num: int):
    """
    Select which segment the AWG is currently playing on a channel.
    (SCPI may need minor adjustment depending on firmware.)
    """
    awg.write(f":TRACe{ch_num}:SELect {seg_num}")


def start_output(awg, mode: int):
    """
    Enable outputs on all channels (safely) and start continuous generation.
    """
    # Continuous initiation
    awg.write(':INITiate:CONTinuous ON')

    # Turn on whatever outputs exist (ignore missing channels)
    for ch in (1, 2, 3, 4):
        try:
            awg.write(f':OUTPut{ch} ON')
        except Exception:
            pass

    # Start generation
    awg.write(':INITiate:IMMediate')
    print("Waveform generation initiated.")


def close_awg(awg):
    """Abort generation and close the VISA session."""
    try:
        awg.write(':ABORt')
    except Exception:
        pass
    awg.close()
