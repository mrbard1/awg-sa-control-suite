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


def _q(sa, cmd, default=""):
    try:
        return sa.query(cmd).strip()
    except Exception as e:
        print(f"Query failed {cmd}: {e}")
        return default


def connect_sa(rm, address, timeout):
    sa = rm.open_resource(address)
    sa.timeout = timeout
    print("SA Identity:", sa.query("*IDN?"))
    print("OPT:", sa.query("*OPT?").strip())
    return sa


def configure_sa(sa, min_freq, max_freq, rbw_hz, vbw_hz=None, points=1001, detector="RMS", avg_on=False, avg_count=20, atten_db=None, preamp_on=None):
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

    # --- Attenuation (SA mode) ---
    if atten_db is not None:
        # Use SA-mode headers consistently
        sa.write(":SENSe:POWer:RF:ATTenuation:AUTO OFF")
        sa.write(f":SENSe:POWer:RF:ATTenuation {float(atten_db)}")
        _q(sa, "*OPC?")

        print("ERR after atten:", _q(sa, ":SYST:ERR?"))
        print("ATTEN auto:", _q(sa, ":SENSe:POWer:RF:ATTenuation:AUTO?"))
        print("ATTEN now :", _q(sa, ":SENSe:POWer:RF:ATTenuation?"))


    # --- Preamplifier (SA mode) ---
    if preamp_on is not None:
        state = "ON" if preamp_on else "OFF"

        # NOTE: Your box threw -113 on GAIN:AUTO, so do NOT send it.
        # sa.write(":SENSe:POWer:RF:GAIN:AUTO OFF")  # <-- remove / unsupported on your firmware

        sa.write(f":SENSe:POWer:RF:GAIN:STATe {state}")
        _q(sa, "*OPC?")

        print("ERR after preamp:", _q(sa, ":SYST:ERR?"))
        print("PREAMP now :", _q(sa, ":SENSe:POWer:RF:GAIN:STATe?"))


    # --- Snapshot (safe queries only) ---
    print("ATTEN (SA hdr):", _q(sa, ":SENSe:POWer:RF:ATTenuation?"))
    print("PREAMP (SA hdr):", _q(sa, ":SENSe:POWer:RF:GAIN:STATe?"))
    print("REF level:", _q(sa, ":DISP:WIND:TRAC:Y:SCAL:RLEV?"))

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
    sa.write(f':DETector {detector}')                  
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



def get_spectrum_data(sa, plot=False, plot_psd=False, plot_both=False, preamp_gain_db=0.0):
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
        settings["preamp_state"] = sa.query("::SENSe:POWer:RF:GAIN:STATe?").strip()
    except Exception:
        settings["preamp_state"] = None
    
    preamp_on = str(settings.get("preamp_state", "")).strip().upper() in ("1", "ON")
    try:
        preamp_gain_db = float(preamp_gain_db)
    except Exception:
        preamp_gain_db = 0.0

    settings["preamp_on"] = preamp_on
    settings["preamp_gain_db"] = preamp_gain_db if preamp_on else 0.0

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
    power_dbmhz_corr = None
    if rbw_hz is not None and np.isfinite(rbw_hz) and rbw_hz > 0:
        power_dbmhz = power_dbm - 10.0 * np.log10(rbw_hz)
        power_dbmhz_corr = power_dbmhz - settings["preamp_gain_db"]

    if plot:
        # Raw power (dBm)
        if plot_both or (not plot_psd):
            plt.figure()
            plt.plot(freq_hz, power_dbm)
            plt.ylabel("Power (dBm)")
            plt.title("Raw Spectrum Analyzer Trace")
            plt.xlabel("Frequency (Hz)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # PSD (dBm/Hz), optionally preamp-corrected
        if (plot_both or plot_psd) and (power_dbmhz_corr is not None):
            plt.figure()
            plt.plot(freq_hz, power_dbmhz_corr)
            plt.ylabel("PSD (dBm/Hz) (preamp-corrected)")
            plt.title("Spectrum (RBW-normalized)")
            plt.xlabel("Frequency (Hz)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()



    return freq_hz, power_data, settings


def close_sa(sa):
    sa.write(':CALibration:AUTO ON')
    sa.close()
    return None
