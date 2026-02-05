# osc_control.py
# Complete oscilloscope control utilities:
# - Connect
# - Configure using user parameters
# - Auto setup
# - Capture and return waveform data
# - Close

import pyvisa
import numpy as np

def connect_osc(rm, address, timeout):
    osc = rm.open_resource(address)
    osc.timeout = timeout
    print("OSC Identity:", osc.query("*IDN?"))
    return osc


def configure_osc(osc, settings):
    """
    Configure oscilloscope using user settings passed from USER_main.py.
    'settings' is a dict, passed in by USER_main.
    """

    # Reset for consistency
    osc.write("*RST")

    # Timebase
    osc.write(f":TIM:SCAL {settings['time_scale']}")
    osc.write(f":TIM:POS {settings['time_position']}")

    # Acquisition mode
    osc.write(f":ACQ:TYPE {settings['acquire_type']}")
    if settings['acquire_type'] == "AVER":
        osc.write(f":ACQ:COUN {settings['acquire_count']}")

    # Channel configuration helper
    def config_channel(ch_num, on, scale):
        if on:
            osc.write(f":CHAN{ch_num}:DISP ON")
            osc.write(f":CHAN{ch_num}:SCAL {scale}")
            osc.write(f":CHAN{ch_num}:COUP {settings['coupling']}")
        else:
            osc.write(f":CHAN{ch_num}:DISP OFF")

    # Configure all four channels
    config_channel(1, settings['ch1_on'], settings['ch1_scale'])
    config_channel(2, settings['ch2_on'], settings['ch2_scale'])
    config_channel(3, settings['ch3_on'], settings['ch3_scale'])
    config_channel(4, settings['ch4_on'], settings['ch4_scale'])

    # Trigger
    osc.write(f":TRIG:EDGE:SOUR {settings['trig_source']}")
    osc.write(f":TRIG:EDGE:LEV {settings['trig_level']}")
    osc.write(f":TRIG:EDGE:SLOP {settings['trig_slope']}")

    # Display
    osc.write(f":DISP:PERS {settings['persistence']}")
    osc.write(f":DISP:GRAD {settings['grid']}")

    print("Oscilloscope configured.\n")


def auto_setup(osc):
    """
    Autoscale + autoconfigure the oscilloscope.
    Useful if the user isn't sure what scale to pick.
    """
    osc.write(":AUToscale")
    print("Oscilloscope autoscaled.")


def capture_display(osc, channel=1):
    """
    Fetch waveform data from the oscilloscope.
    Returns (time_vector, voltage_vector).
    """

    # Set up waveform format
    osc.write(":WAV:FORM BYTE")
    osc.write(":WAV:MODE RAW")
    osc.write(f":WAV:SOUR CHAN{channel}")

    # Query preamble for scaling info
    pre = osc.query(":WAV:PRE?").split(',')
    # Keysight/MSO-X style parsing:
    # pre[7] = x increment, pre[8] = x origin, pre[9] = x reference
    # pre[12] = y increment, etc.
    x_inc = float(pre[7])
    x_origin = float(pre[8])
    y_inc = float(pre[12])
    y_origin = float(pre[13])
    y_ref = float(pre[14])

    # Binary fetch
    raw = osc.query_binary_values(":WAV:DATA?", datatype='B')

    raw = np.array(raw)
    voltage = (raw - y_ref) * y_inc + y_origin
    time = x_origin + np.arange(len(raw)) * x_inc

    return time, voltage


def close_osc(osc):
    osc.close()
