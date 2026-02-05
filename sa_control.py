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


def configure_sa(sa, min_freq, max_freq, steps):
    print("This spectrum analyzer collection has not been completely configured!")

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
    step_mhz = steps / 1e6

    sa.write(':INST: SA')
    sa.write(f':FREQuency:CENTer {cent_ghz}GHz')    # Center frequency
    sa.write(f':FREQuency:SPAN {span_mhz}MHz')      # Frequency span
    sa.write(f':BANDwidth {step_mhz}MHz')           # Resolution bandwidth
    sa.write(':BANDwidth:VIDeo:AUTO')               # Video bandwidth auto
    sa.write(':SWEep:POINts 1001')                  # Points per sweep
    sa.write(':UNIT:POWer DBM')                     # dBm power units
    sa.write(':TRACe:TYPE WRITe')                   # Trace type
    sa.write(':DETector POSitive')                  # Positive peak detection
    sa.write(':INITiate:CONTinuous OFF')            # Single-sweep mode


def initial_sweep(sa):
    print("This spectrum analyzer collection has not been completely configured!")

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

    if plot:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(freq_hz/1e6, power_data)
        plt.tight_layout()
        plt.title('Arbitrary Spectrum Analyzer Readout')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dBm)')
        plt.grid(True)
        plt.show()

    return freq_hz, power_data


def close_sa(sa):
    sa.write(':CALibration:AUTO ON')
    sa.close()
    return None
