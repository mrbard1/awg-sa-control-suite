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


def configure_sa(sa, min_freq, max_freq, rbw_hz, vbw_hz=None, points=1001, detector="RMS", avg_on=False, avg_count=20):
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
    sa.write(':DETector {detector}')                  
    sa.write(':INITiate:CONTinuous OFF')            # Single-sweep mode
        # Averaging
    if avg_on:
        sa.write(':AVERage:STATe ON')
        sa.write(f':AVERage:COUNt {int(avg_count)}')
        try:
            sa.write(':AVERage:CLEar')  # clear previous averages if supported
        except Exception:
            pass
    else:
        sa.write(':AVERage:STATe OFF')
          


def initial_sweep(sa, sweeps=1):
    # Trigger sweep(s) and wait for completion
    for _ in range(int(sweeps)):
        sa.write(':INITiate:IMMediate')
        sa.query('*OPC?')
    print("Read Completed!\n")



def get_spectrum_data(sa, plot=False, plot_psd=False):
    power_data = sa.query(':TRAC? TRACE1')
    power_data = np.array([float(x) for x in power_data.split(',')], dtype=float)
    power_dbm = power_data

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
    # --- PSD (dBm/Hz) derived from RBW ---
    rbw_hz = settings.get("rbw_hz", None)
    if rbw_hz is not None:
        try:
            rbw_hz = float(rbw_hz)
        except Exception:
            rbw_hz = None

    power_dbmhz = None
    if rbw_hz is not None and np.isfinite(rbw_hz) and rbw_hz > 0:
        power_dbmhz = power_dbm - 10.0 * np.log10(rbw_hz)

    if plot:
        plt.figure()
        if plot_psd and (power_dbmhz is not None):
            plt.plot(freq_hz, power_dbmhz)
            plt.ylabel("PSD (dBm/Hz)")
            plt.title("Spectrum (RBW-normalized)")
        else:
            plt.plot(freq_hz, power_dbm)
            plt.ylabel("Power (dBm)")
            plt.title("Spectrum Analyzer Trace")

        plt.xlabel("Frequency (Hz)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    return freq_hz, power_data, settings


def close_sa(sa):
    sa.write(':CALibration:AUTO ON')
    sa.close()
    return None
