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
import csv
import os
import tempfile


TOPOLOGY_GAIN_DB_DIFFERENTIAL = 6.0206


def _interp1d_clamped(x, xp, fp):
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if xp.size == 0 or fp.size == 0 or xp.size != fp.size:
        raise ValueError("Invalid interpolation table: xp/fp must be same non-zero length.")
    if xp.size == 1:
        return float(fp[0])
    x = float(np.clip(float(x), float(xp[0]), float(xp[-1])))
    return float(np.interp(x, xp, fp))


class CwCalTable:
    """
    CW correction lookup table from CSV:
      freq_hz, amp_vpk_cmd, corr_db

    Query behavior:
    - Clamp query to min/max freq and amp.
    - If table is a full rectangular grid, use bilinear interpolation.
    - Else fallback to 1D interpolation over freq at nearest amp.
    - Final fallback is nearest-neighbor over all points.
    """
    REQUIRED_COLS = ("freq_hz", "amp_vpk_cmd", "corr_db")

    def __init__(self, csv_path):
        if not csv_path:
            raise ValueError("CW calibration CSV path is empty.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CW calibration CSV not found: {csv_path}")
        self.csv_path = csv_path
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Malformed CW calibration CSV (no header): {csv_path}")
            missing = [c for c in self.REQUIRED_COLS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"CW calibration CSV missing required columns {missing}: {csv_path}"
                )
            rows = []
            for i, row in enumerate(reader, start=2):
                try:
                    rows.append(
                        (
                            float(row["freq_hz"]),
                            float(row["amp_vpk_cmd"]),
                            float(row["corr_db"]),
                        )
                    )
                except Exception as e:
                    raise ValueError(f"Invalid CW calibration row at line {i}: {e}") from e

        if not rows:
            raise ValueError(f"CW calibration CSV has no data rows: {csv_path}")

        # Aggregate duplicates by mean correction for robustness.
        agg = {}
        for f, a, c in rows:
            agg.setdefault((f, a), []).append(c)
        self.points = np.array([(k[0], k[1], float(np.mean(v))) for k, v in agg.items()], dtype=float)

        self.freqs = np.array(sorted({p[0] for p in self.points}), dtype=float)
        self.amps = np.array(sorted({p[1] for p in self.points}), dtype=float)
        self._grid = np.full((self.freqs.size, self.amps.size), np.nan, dtype=float)
        self._pair_to_corr = {}
        for f, a, c in self.points:
            fi = int(np.where(self.freqs == f)[0][0])
            ai = int(np.where(self.amps == a)[0][0])
            self._grid[fi, ai] = c
            self._pair_to_corr[(f, a)] = c

        self.full_grid = bool(np.all(np.isfinite(self._grid)))

    def _nearest_freq_amp(self, freq_hz, amp_vpk_cmd):
        f = float(np.clip(freq_hz, self.freqs[0], self.freqs[-1]))
        a = float(np.clip(amp_vpk_cmd, self.amps[0], self.amps[-1]))
        fi = int(np.argmin(np.abs(self.freqs - f)))
        ai = int(np.argmin(np.abs(self.amps - a)))
        return float(self.freqs[fi]), float(self.amps[ai])

    def _bilinear(self, freq_hz, amp_vpk_cmd):
        f = float(np.clip(freq_hz, self.freqs[0], self.freqs[-1]))
        a = float(np.clip(amp_vpk_cmd, self.amps[0], self.amps[-1]))
        fi_hi = int(np.searchsorted(self.freqs, f, side="left"))
        ai_hi = int(np.searchsorted(self.amps, a, side="left"))

        fi0 = max(0, min(fi_hi - 1, self.freqs.size - 1))
        fi1 = max(0, min(fi_hi, self.freqs.size - 1))
        ai0 = max(0, min(ai_hi - 1, self.amps.size - 1))
        ai1 = max(0, min(ai_hi, self.amps.size - 1))

        f0, f1 = self.freqs[fi0], self.freqs[fi1]
        a0, a1 = self.amps[ai0], self.amps[ai1]
        c00 = self._grid[fi0, ai0]
        c01 = self._grid[fi0, ai1]
        c10 = self._grid[fi1, ai0]
        c11 = self._grid[fi1, ai1]

        if fi0 == fi1 and ai0 == ai1:
            return float(c00)
        if fi0 == fi1:
            return float(np.interp(a, [a0, a1], [c00, c01]))
        if ai0 == ai1:
            return float(np.interp(f, [f0, f1], [c00, c10]))

        t = (f - f0) / (f1 - f0)
        u = (a - a0) / (a1 - a0)
        return float((1 - t) * (1 - u) * c00 + (1 - t) * u * c01 + t * (1 - u) * c10 + t * u * c11)

    def _fallback(self, freq_hz, amp_vpk_cmd):
        # 1D interpolation across freq at nearest amp, if enough points exist.
        _, a_near = self._nearest_freq_amp(freq_hz, amp_vpk_cmd)
        mask = np.isclose(self.points[:, 1], a_near)
        subset = self.points[mask]
        if subset.shape[0] >= 2:
            idx = np.argsort(subset[:, 0])
            return _interp1d_clamped(freq_hz, subset[idx, 0], subset[idx, 2])
        if subset.shape[0] == 1:
            return float(subset[0, 2])

        # Final fallback: nearest-neighbor in (freq, amp).
        f = float(np.clip(freq_hz, self.freqs[0], self.freqs[-1]))
        a = float(np.clip(amp_vpk_cmd, self.amps[0], self.amps[-1]))
        d2 = (self.points[:, 0] - f) ** 2 + (self.points[:, 1] - a) ** 2
        return float(self.points[int(np.argmin(d2)), 2])

    def corr_db(self, freq_hz, amp_vpk_cmd):
        if self.full_grid:
            return self._bilinear(freq_hz, amp_vpk_cmd)
        return self._fallback(freq_hz, amp_vpk_cmd)


class CombinerILTable:
    """Combiner insertion-loss table from CSV columns: freq_hz, il_db."""
    REQUIRED_COLS = ("freq_hz", "il_db")

    def __init__(self, csv_path):
        if not csv_path:
            raise ValueError("Combiner IL CSV path is empty.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Combiner IL CSV not found: {csv_path}")
        self.csv_path = csv_path
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Malformed combiner IL CSV (no header): {csv_path}")
            missing = [c for c in self.REQUIRED_COLS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"Combiner IL CSV missing required columns {missing}: {csv_path}"
                )
            rows = []
            for i, row in enumerate(reader, start=2):
                try:
                    rows.append((float(row["freq_hz"]), float(row["il_db"])))
                except Exception as e:
                    raise ValueError(f"Invalid combiner IL row at line {i}: {e}") from e
        if not rows:
            raise ValueError(f"Combiner IL CSV has no data rows: {csv_path}")

        agg = {}
        for f, il in rows:
            agg.setdefault(f, []).append(il)
        items = sorted((f, float(np.mean(v))) for f, v in agg.items())
        self.freqs = np.array([x[0] for x in items], dtype=float)
        self.ils = np.array([x[1] for x in items], dtype=float)

    def il_db(self, freq_hz):
        return _interp1d_clamped(freq_hz, self.freqs, self.ils)


class MeasurementCalibration:
    """
    Applies measurement-path correction in dB:
      P_corrected = P_measured + corr_db

    Convention:
    - amp_vpk_cmd is always PER-LEG Vpk.
    - differential_combined adds +6.0206 dB topology gain.
    - if combiner IL is enabled, subtract IL(f) from correction.
    """
    def __init__(
        self,
        output_topology="single_leg",
        output_leg="DATA",
        amp_vpk_cmd=None,
        cw_table=None,
        combiner_enabled=False,
        il_table=None,
        cw_cal_csv_path=None,
        il_csv_path=None,
    ):
        topo = str(output_topology or "single_leg").strip().lower()
        if topo not in ("single_leg", "differential_combined"):
            raise ValueError(
                "Invalid calibration.output_topology. Expected 'single_leg' or 'differential_combined'."
            )
        leg = str(output_leg or "DATA").strip().upper()
        if topo == "single_leg" and leg not in ("DATA", "DATA_BAR"):
            raise ValueError("calibration.output_leg must be DATA or DATA_BAR for single_leg topology.")

        self.output_topology = topo
        self.output_leg = leg
        self.amp_vpk_cmd = None if amp_vpk_cmd is None else float(amp_vpk_cmd)
        self.cw_table = cw_table
        self.combiner_enabled = bool(combiner_enabled)
        self.il_table = il_table
        self.cw_cal_csv_path = cw_cal_csv_path
        self.il_csv_path = il_csv_path
        self.topology_gain_db = TOPOLOGY_GAIN_DB_DIFFERENTIAL if topo == "differential_combined" else 0.0

    def corr_db(self, freq_hz):
        total = self.topology_gain_db
        if self.cw_table is not None:
            if self.amp_vpk_cmd is None:
                raise ValueError("amp_vpk_cmd is required when cw_cal_csv_path is configured.")
            total += float(self.cw_table.corr_db(freq_hz, self.amp_vpk_cmd))
        if self.combiner_enabled:
            if self.il_table is None:
                raise ValueError("combiner.enabled=true but combiner IL table is not loaded.")
            total -= float(self.il_table.il_db(freq_hz))
        return float(total)


def run_calibration_self_checks():
    """Minimal unit-style checks for CSV parsing and interpolation behavior."""
    with tempfile.TemporaryDirectory() as td:
        cw_csv = os.path.join(td, "cw.csv")
        with open(cw_csv, "w", newline="") as f:
            f.write("freq_hz,amp_vpk_cmd,corr_db\n")
            f.write("100,0.1,1.0\n")
            f.write("200,0.1,3.0\n")
            f.write("100,0.2,5.0\n")
            f.write("200,0.2,7.0\n")
        cw = CwCalTable(cw_csv)
        assert cw.full_grid, "Expected full-grid calibration table."
        assert abs(cw.corr_db(150.0, 0.15) - 4.0) < 1e-9, "Bilinear interpolation failed."
        assert abs(cw.corr_db(1.0, 1.0) - 5.0) < 1e-9, "Clamping failed for CW table."

        il_csv = os.path.join(td, "il.csv")
        with open(il_csv, "w", newline="") as f:
            f.write("freq_hz,il_db\n")
            f.write("100,1.0\n")
            f.write("200,3.0\n")
        il = CombinerILTable(il_csv)
        assert abs(il.il_db(150.0) - 2.0) < 1e-9, "1D interpolation failed for IL table."

        mc = MeasurementCalibration(
            output_topology="differential_combined",
            output_leg="DATA",
            amp_vpk_cmd=0.15,
            cw_table=cw,
            combiner_enabled=True,
            il_table=il,
        )
        expected = TOPOLOGY_GAIN_DB_DIFFERENTIAL + 4.0 - 2.0
        assert abs(mc.corr_db(150.0) - expected) < 1e-9, "Combined correction computation failed."


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



def get_spectrum_data(
    sa,
    plot=False,
    plot_psd=False,
    plot_both=False,
    preamp_gain_db=0.0,
    measurement_cal=None,
    return_raw_and_corrected=False,
):
    power_data = sa.query(':TRAC? TRACE1')
    power_data = np.array([float(x) for x in power_data.split(',')], dtype=float)
    power_dbm_raw = power_data.copy()

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
    corr_db = np.zeros_like(power_dbm_raw, dtype=float)
    if measurement_cal is not None:
        corr_db = np.array([measurement_cal.corr_db(ff) for ff in freq_hz], dtype=float)
    power_dbm = power_dbm_raw + corr_db
    settings["cal_corr_db_mean"] = float(np.mean(corr_db)) if corr_db.size else 0.0
    settings["cal_corr_db_min"] = float(np.min(corr_db)) if corr_db.size else 0.0
    settings["cal_corr_db_max"] = float(np.max(corr_db)) if corr_db.size else 0.0
    settings["calibration_applied"] = bool(measurement_cal is not None)

    rbw_hz = settings.get("rbw_hz", None)
    if rbw_hz is not None:
        try:
            rbw_hz = float(rbw_hz)
        except Exception:
            rbw_hz = None

    power_dbmhz = None
    power_dbmhz_corr = None
    if rbw_hz is not None and np.isfinite(rbw_hz) and rbw_hz > 0:
        # Path calibration is independent of RBW; apply once before RBW normalization.
        power_dbmhz = power_dbm - 10.0 * np.log10(rbw_hz)
        power_dbmhz_corr = power_dbmhz - settings["preamp_gain_db"]

    if plot:
        # Raw power (dBm)
        if plot_both or (not plot_psd):
            plt.figure()
            plt.plot(freq_hz, power_dbm)
            plt.ylabel("Power (dBm)")
            plt.title("Spectrum Analyzer Trace (Corrected)")
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



    if return_raw_and_corrected:
        return freq_hz, power_dbm_raw, power_dbm, settings
    return freq_hz, power_dbm, settings


def close_sa(sa):
    sa.write(':CALibration:AUTO ON')
    sa.close()
    return None
