# sa_control.py
# Spectrum analyzer control:
#   - Connect
#   - Reset and configuration
#   - Trigger a sweep
#   - Fetch and plot spectrum
#   - Close (re-enable auto calibration)

import matplotlib.pyplot as plt
import numpy as np
import pyvisa
import time


def connect_sa(rm, address, timeout):
    sa = rm.open_resource(address)
    sa.timeout = timeout
    print("SA Identity:", sa.query("*IDN?"))
    return sa


def configure_sa(sa, min_freq, max_freq, rbw_hz, vbw_hz=None, points=1001):
    print("Sending *RST...")
    sa.write('*RST')
    print("Waiting for reset to complete...")
    time.sleep(5)  # Increased wait time for reset

    # Check if reset completed
    print("Checking operation complete...")
    opc_response = sa.query('*OPC?')
    print(f"*OPC? response: {opc_response.strip()}")

    # Check for errors after reset
    error = sa.query(':SYST:ERR?')
    print(f"System error after *RST: {error.strip()}")
    if error.strip() != '+0,"No error"':
        raise Exception(f"Instrument error after *RST: {error}")

    center = (min_freq + max_freq) / 2
    cent_ghz = center / 1e9
    span = max_freq - min_freq
    span_mhz = span / 1e6
    sa.write(':INST: SA')
    sa.write(f':FREQuency:CENTer {cent_ghz}GHz')    # Center frequency
    sa.write(f':FREQuency:SPAN {span_mhz}MHz')      # Frequency span

    # RBW/VBW (Hz). rbw_hz is REQUIRED; vbw_hz optional (None => AUTO)
    sa.write(f':BANDwidth {float(rbw_hz)}Hz')       # Resolution bandwidth
    if vbw_hz is None:
        sa.write(':BANDwidth:VIDeo:AUTO ON')
    else:
        sa.write(':BANDwidth:VIDeo:AUTO OFF')
        sa.write(f':BANDwidth:VIDeo {float(vbw_hz)}Hz')

    sa.write(f':SWEep:POINts {int(points)}')        # Points per sweep
    sa.write(':UNIT:POWer DBM')                     # dBm power units
    sa.write(':TRACe:TYPE WRITe')                   # Trace type
    sa.write(':DETector POSitive')                  # Positive peak detection
    sa.write(':INITiate:CONTinuous OFF')            # Single-sweep mode


def initial_sweep(sa):
    # Trigger a single sweep and wait for completion.
    sa.write(':INITiate:IMMediate')  # Trigger sweep
    completed = sa.query('*OPC?')    # Wait for sweep to finish
    print("Read Completed!\n")


def get_spectrum_data(sa, plot=False):
    power_data = sa.query(':TRAC? TRACE1')
    power_data = np.array([float(x) for x in power_data.split(',')], dtype=float)

    start_freq = float(sa.query(':FREQuency:STARt?'))  # Hz
    stop_freq  = float(sa.query(':FREQuency:STOP?'))   # Hz
    num_points = int(float(sa.query(':SWEep:POINts?')))

    freq_hz = np.linspace(start_freq, stop_freq, num_points)
    # --- Settings snapshot (required for RBW-aware PSD) ---
    settings = {}

    # Frequencies / sweep
    settings["start_freq_hz"] = start_freq
    settings["stop_freq_hz"] = stop_freq
    settings["num_points"] = num_points

    # RBW / VBW / detector / averaging / attenuation / preamp / ref level
    try:
        settings["rbw_hz"] = float(sa.query(":BANDwidth?"))
    except Exception:
        settings["rbw_hz"] = None

    try:
        vbw_auto = sa.query(":BANDwidth:VIDeo:AUTO?").strip()
        settings["vbw_auto"] = vbw_auto
        if vbw_auto in ("1", "ON"):
            settings["vbw_hz"] = None
        else:
            settings["vbw_hz"] = float(sa.query(":BANDwidth:VIDeo?"))
    except Exception:
        settings["vbw_auto"] = None
        settings["vbw_hz"] = None

    try:
        settings["detector"] = sa.query(":DETector?").strip()
    except Exception:
        settings["detector"] = None

    try:
        settings["avg_state"] = sa.query(":AVERage:STATe?").strip()
        settings["avg_count"] = int(float(sa.query(":AVERage:COUNt?")))
    except Exception:
        settings["avg_state"] = None
        settings["avg_count"] = None

    try:
        settings["atten_db"] = float(sa.query(":POWer:ATTenuation?"))
    except Exception:
        settings["atten_db"] = None

    try:
        settings["preamp_state"] = sa.query(":POWer:GAIN:STATe?").strip()
    except Exception:
        settings["preamp_state"] = None

    try:
        settings["ref_level_dbm"] = float(sa.query(":DISPlay:WINDow:TRACe:Y:SCALe:RLEVel?"))
    except Exception:
        settings["ref_level_dbm"] = None
 
    if plot:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(freq_hz/1e6, power_data)
        plt.tight_layout()
        plt.title('Arbitrary Spectrum Analyzer Readout')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dBm)')
        plt.grid(True)
        plt.show()

    return freq_hz, power_data, settings


def close_sa(sa):
    sa.write(':CALibration:AUTO ON')
    sa.close()
    return None
