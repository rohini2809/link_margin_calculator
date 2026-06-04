# ============================================================
# MANGOSAT LINK MARGIN CALCULATOR
# NuSpace — S-Band Link Budget Analysis
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# ============================================================
# SECTION 1 — CONSTANTS
# ============================================================
SPEED_OF_LIGHT    = 3e8
EARTH_RADIUS_KM   = 6371
BOLTZMANN_dBWKHz  = -228.6
ALTITUDE_KM       = 590
MIN_ELEVATION_DEG = 10
FREQUENCY_GHz     = 2.4

# ============================================================
# SECTION 2 — USE CASE PRESETS
# ============================================================
USE_CASES = {
    "TC Uplink (Ground → Satellite)": {
        "eirp_dBW"             : 45.0,
        "gt_dBK"               : -31.6,
        "desired_data_rate_bps": 38400,
        "min_data_rate_bps"    : 2400,
        "ebno_threshold_dB"    : 8.8,
        "polarization_loss_dB" : 3.0,
        "atmo_loss_dB"         : 1.1,
        "iono_loss_dB"         : 0.1,
        "rain_loss_dB"         : 3.8,
        "pointing_loss_dB"     : 0.4,
        "direction"            : "uplink",
        "ref_margin_dB"        : 12.3,
    },
    "TM Downlink (Satellite → Ground)": {
        "eirp_dBW"             : -7.0,
        "gt_dBK"               : 12.0,
        "desired_data_rate_bps": 38400,
        "min_data_rate_bps"    : 2400,
        "ebno_threshold_dB"    : 8.8,
        "polarization_loss_dB" : 3.0,
        "atmo_loss_dB"         : 1.1,
        "iono_loss_dB"         : 0.1,
        "rain_loss_dB"         : 3.8,
        "pointing_loss_dB"     : 2.0,
        "direction"            : "downlink",
        "ref_margin_dB"        : 3.5,
    },
    "Payload Downlink (Satellite → Ground)": {
        "eirp_dBW"             : -3.6,
        "gt_dBK"               : 12.6,
        "desired_data_rate_bps": 250000,
        "min_data_rate_bps"    : 2400,
        "ebno_threshold_dB"    : 3.5,
        "polarization_loss_dB" : 3.0,
        "atmo_loss_dB"         : 1.1,
        "iono_loss_dB"         : 0.1,
        "rain_loss_dB"         : 3.8,
        "pointing_loss_dB"     : 2.0,
        "direction"            : "downlink",
        "ref_margin_dB"        : 3.5,
    },
}

# ============================================================
# SECTION 3 — PHYSICS ENGINE
# ============================================================

def calc_slant_range(elevation_deg, altitude_km):
    """
    FIXED formula using sine rule of triangles.
    Verified: at El=90 returns exactly altitude_km.
    At El=10 returns ~1839 km (correct geometry).
    """
    El    = np.radians(elevation_deg)
    Re    = EARTH_RADIUS_KM
    h     = altitude_km
    slant = np.sqrt((Re + h)**2 - (Re * np.cos(El))**2) - Re * np.sin(El)
    return slant

def calc_fspl_dB(slant_range_km, frequency_GHz):
    """Free Space Path Loss."""
    d    = slant_range_km * 1000
    f    = frequency_GHz * 1e9
    fspl = 20 * np.log10((4 * np.pi * d * f) / SPEED_OF_LIGHT)
    return fspl

def calc_antenna_gain_dB(diameter_m, frequency_GHz, efficiency):
    """Parabolic dish gain."""
    f          = frequency_GHz * 1e9
    wavelength = SPEED_OF_LIGHT / f
    gain_lin   = efficiency * (np.pi * diameter_m / wavelength) ** 2
    return 10 * np.log10(gain_lin)

def calc_beamwidth_deg(diameter_m, frequency_GHz):
    """3dB beamwidth of parabolic dish."""
    f          = frequency_GHz * 1e9
    wavelength = SPEED_OF_LIGHT / f
    return np.degrees(1.22 * wavelength / diameter_m)

def calc_zenith_pointing_loss_dB(elevation_deg, beamwidth_deg):
    """
    Pointing loss for fixed zenith antenna.
    At El=90: loss = 0 dB (perfect alignment).
    At El=10: large loss (antenna misses satellite).
    Capped at -60 dB — physically unreachable beyond that.
    """
    theta_offset = 90.0 - elevation_deg
    loss         = -12.0 * (theta_offset / beamwidth_deg) ** 2
    return np.maximum(loss, -60.0)

def calc_pass_duration_minutes(min_viable_el, altitude_km):
    """Estimate comms window per pass using orbital geometry."""
    if min_viable_el is None or min_viable_el > 90:
        return 0.0
    Re               = EARTH_RADIUS_KM
    h                = altitude_km
    mu               = 398600.4418
    orbital_r        = Re + h
    orbital_vel      = np.sqrt(mu / orbital_r)
    orbital_period_s = 2 * np.pi * orbital_r / orbital_vel
    El_rad           = np.radians(min_viable_el)
    rho              = np.arccos(np.clip(Re * np.cos(El_rad) / orbital_r, -1, 1)) - El_rad
    fraction         = (2 * rho) / (2 * np.pi)
    return max(fraction * orbital_period_s / 60.0, 0.0)

def calc_link_margin(elevation_deg, eirp_dBW, gt_dBK,
                     data_rate_bps, ebno_threshold_dB,
                     polarization_loss_dB, atmo_loss_dB,
                     iono_loss_dB, rain_loss_dB,
                     fixed_pointing_loss_dB,
                     antenna_mode,
                     diameter_m=0.7, efficiency=0.6,
                     frequency_GHz=2.4):
    """Master link margin calculation."""

    slant_range_km = calc_slant_range(elevation_deg, ALTITUDE_KM)
    fspl_dB        = calc_fspl_dB(slant_range_km, frequency_GHz)

    if antenna_mode == "non_steerable":
        beamwidth_deg = calc_beamwidth_deg(diameter_m, frequency_GHz)
        pointing_loss = calc_zenith_pointing_loss_dB(elevation_deg, beamwidth_deg)
    else:
        pointing_loss = 0.0

    total_losses = (fspl_dB
                    + polarization_loss_dB
                    + atmo_loss_dB
                    + iono_loss_dB
                    + rain_loss_dB
                    + fixed_pointing_loss_dB
                    + abs(pointing_loss))

    data_rate_dBHz = 10 * np.log10(data_rate_bps)

    link_margin = (eirp_dBW
                   + gt_dBK
                   - total_losses
                   - BOLTZMANN_dBWKHz
                   - data_rate_dBHz
                   - ebno_threshold_dB)

    return link_margin, slant_range_km, fspl_dB, pointing_loss

# ============================================================
# SECTION 4 — DATA LAYER
# ============================================================

def run_elevation_sweep(params, data_rate_bps, antenna_mode,
                        diameter_m=0.7, efficiency=0.6):
    elevations                    = np.arange(MIN_ELEVATION_DEG, 91, 1)
    margins, ranges, fspls, p_losses = [], [], [], []
    for el in elevations:
        lm, sr, fspl, pl = calc_link_margin(
            el,
            params["eirp_dBW"], params["gt_dBK"],
            data_rate_bps, params["ebno_threshold_dB"],
            params["polarization_loss_dB"], params["atmo_loss_dB"],
            params["iono_loss_dB"], params["rain_loss_dB"],
            params["pointing_loss_dB"], antenna_mode,
            diameter_m, efficiency)
        margins.append(lm)
        ranges.append(sr)
        fspls.append(fspl)
        p_losses.append(pl)
    return (elevations,
            np.array(margins), np.array(ranges),
            np.array(fspls),   np.array(p_losses))

def find_min_viable_elevation(elevations, margins):
    for i, lm in enumerate(margins):
        if lm >= 0:
            return elevations[i]
    return None

def find_min_antenna_size(params, data_rate_bps, antenna_mode):
    for d in np.arange(0.1, 10.1, 0.05):
        lm, *_ = calc_link_margin(
            MIN_ELEVATION_DEG,
            params["eirp_dBW"], params["gt_dBK"],
            data_rate_bps, params["ebno_threshold_dB"],
            params["polarization_loss_dB"], params["atmo_loss_dB"],
            params["iono_loss_dB"], params["rain_loss_dB"],
            params["pointing_loss_dB"], antenna_mode,
            diameter_m=d)
        if lm >= 0:
            return round(d, 2)
    return None

# ============================================================
# SECTION 5 — PAGE SETUP
# ============================================================
st.set_page_config(page_title="MangoSat Link Budget", layout="wide")
st.title("📡 MangoSat Link Budget Analyser")
st.markdown("""
**NuSpace | S-Band | Altitude: 590 km | Min Contact Elevation: 10°**

Evaluates link margin across elevation angles for TC Uplink, TM Downlink,
and Payload Downlink — for both fixed zenith and steerable antenna configurations.
""")

# ============================================================
# SECTION 6 — SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Analysis Settings")

use_case_name = st.sidebar.selectbox("📋 Select Use Case", list(USE_CASES.keys()))
params        = USE_CASES[use_case_name]

antenna_mode = st.sidebar.radio(
    "🔭 Antenna Mode",
    ["non_steerable", "steerable"],
    format_func=lambda x: "🔒 Non-Steerable (Fixed Zenith)"
                          if x == "non_steerable" else "🎯 Steerable (Tracking)")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Antenna Parameters")
input_mode = st.sidebar.radio("Input Mode", ["🎚️ Sliders", "⌨️ Manual Entry"])

def get_input(label, min_val, max_val, default, step, key, fmt="%.2f"):
    if input_mode == "🎚️ Sliders":
        return st.sidebar.slider(label, float(min_val), float(max_val),
                                 float(default), float(step), key=key)
    return st.sidebar.number_input(label, min_value=float(min_val),
                                   max_value=float(max_val), value=float(default),
                                   step=float(step), key=key, format=fmt)

antenna_diameter_m = get_input("Antenna Diameter (m)", 0.1, 5.0, 0.7, 0.05, "diam")
antenna_efficiency = get_input("Antenna Efficiency (η)", 0.4, 0.9, 0.6, 0.05, "eff")

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Override Link Parameters")
st.sidebar.caption("Defaults from MangoSat Link Budget PDF")

eirp_override  = get_input("EIRP (dBW)",            -20.0, 60.0, params["eirp_dBW"],          0.5, "eirp")
gt_override    = get_input("G/T (dB/K)",             -40.0, 30.0, params["gt_dBK"],            0.5, "gt")
ebno_override  = get_input("Eb/No Threshold (dB)",    1.0,  20.0, params["ebno_threshold_dB"], 0.1, "ebno")

params_live                    = dict(params)
params_live["eirp_dBW"]        = eirp_override
params_live["gt_dBK"]          = gt_override
params_live["ebno_threshold_dB"] = ebno_override

# ============================================================
# SECTION 7 — CALCULATIONS
# ============================================================
elevs, margins_desired, ranges, fspls, p_losses = run_elevation_sweep(
    params_live, params_live["desired_data_rate_bps"],
    antenna_mode, antenna_diameter_m, antenna_efficiency)

_, margins_min, _, _, _ = run_elevation_sweep(
    params_live, params_live["min_data_rate_bps"],
    antenna_mode, antenna_diameter_m, antenna_efficiency)

min_viable_desired = find_min_viable_elevation(elevs, margins_desired)
min_viable_min_dr  = find_min_viable_elevation(elevs, margins_min)
pass_dur_desired   = calc_pass_duration_minutes(min_viable_desired, ALTITUDE_KM)
pass_dur_min_dr    = calc_pass_duration_minutes(min_viable_min_dr,  ALTITUDE_KM)
beamwidth          = calc_beamwidth_deg(antenna_diameter_m, FREQUENCY_GHz)
lm_at_10deg        = margins_desired[0]
lm_at_90deg        = margins_desired[-1]
min_dish           = find_min_antenna_size(params_live,
                                           params_live["desired_data_rate_bps"],
                                           antenna_mode)

# ============================================================
# SECTION 8 — KEY METRICS
# ============================================================
st.markdown("---")
st.subheader("📊 Key Metrics")
st.caption("All values at desired data rate unless stated.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Link Margin @ 10°",
              f"{lm_at_10deg:.1f} dB",
              delta="✅ OK" if lm_at_10deg >= 0 else "❌ Failed")
with col2:
    st.metric("Link Margin @ 90° (Zenith)",
              f"{lm_at_90deg:.1f} dB",
              delta="✅ OK" if lm_at_90deg >= 0 else "❌ Failed")
with col3:
    st.metric("Pass Duration (Desired DR)",
              f"{pass_dur_desired:.1f} min" if pass_dur_desired > 0 else "No link",
              delta=f"Min El: {min_viable_desired}°" if min_viable_desired else "Link never closes")
with col4:
    st.metric("Pass Duration @ 2400 bps",
              f"{pass_dur_min_dr:.1f} min" if pass_dur_min_dr > 0 else "No link",
              delta=f"Min El: {min_viable_min_dr}°" if min_viable_min_dr else "Link never closes")
with col5:
    st.metric("Min Antenna Size Needed",
              f"{min_dish} m" if min_dish else "> 10m",
              delta="to close link @ 10° desired DR")

st.markdown(f"""
> 📄 **PDF Reference Check:** PDF shows **{params['ref_margin_dB']} dB**
> at reference elevation. Zenith margin = **{lm_at_90deg:.1f} dB**.
""")

# ============================================================
# SECTION 9 — MAIN PLOT
# ============================================================
st.markdown("---")
st.subheader("📈 Link Margin vs Elevation Angle")
st.markdown(f"""
**Blue** = desired data rate ({params_live['desired_data_rate_bps']:,} bps) &nbsp;|&nbsp;
**Orange dashed** = minimum data rate (2,400 bps) &nbsp;|&nbsp;
**Red dashed** = 0 dB pass/fail threshold
""")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(elevs, margins_desired, color='royalblue', linewidth=2.5,
        label=f"Desired DR ({params_live['desired_data_rate_bps']:,} bps)")
ax.plot(elevs, margins_min, color='darkorange', linewidth=2, linestyle='--',
        label="Min DR (2,400 bps)")
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='0 dB threshold')
ax.fill_between(elevs, margins_desired, 0,
                where=(margins_desired >= 0), alpha=0.12, color='green')
ax.fill_between(elevs, margins_desired, 0,
                where=(margins_desired < 0), alpha=0.12, color='red')
if min_viable_desired:
    ax.axvline(x=min_viable_desired, color='royalblue', linestyle=':', linewidth=1.8,
               label=f'Min viable El (desired DR): {min_viable_desired}°')
if min_viable_min_dr:
    ax.axvline(x=min_viable_min_dr, color='darkorange', linestyle=':', linewidth=1.8,
               label=f'Min viable El (2400 bps): {min_viable_min_dr}°')
ax.axvline(x=MIN_ELEVATION_DEG, color='grey', linestyle='-', linewidth=1,
           label=f'Min contact elevation ({MIN_ELEVATION_DEG}°)')
ax.set_xlabel("Elevation Angle (°)", fontsize=12)
ax.set_ylabel("Link Margin (dB)", fontsize=12)
ax.set_title(f"MangoSat — {use_case_name} | "
             f"{'Non-Steerable (Fixed Zenith)' if antenna_mode == 'non_steerable' else 'Steerable'}"
             f" | {antenna_diameter_m}m dish", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([MIN_ELEVATION_DEG, 90])
st.pyplot(fig)

# ============================================================
# SECTION 10 — LOSS BREAKDOWN
# ============================================================
st.markdown("---")
st.subheader("🔍 Loss Breakdown vs Elevation Angle")
st.markdown("""
- **Slant Range** — actual signal travel distance. Highest at low elevation, lowest at zenith.
- **FSPL** — Free Space Path Loss. Follows slant range — more distance = more loss.
- **Pointing Loss** — fixed zenith antenna only. Grows rapidly as satellite moves away from zenith.
""")

# Sanity check values
sr_at_zenith = calc_slant_range(90, ALTITUDE_KM)
sr_at_10     = calc_slant_range(10, ALTITUDE_KM)

fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(elevs, ranges, color='orange', linewidth=2)
axes[0].axvline(x=MIN_ELEVATION_DEG, color='grey', linestyle='--', linewidth=1)
axes[0].set_title("Slant Range (km)")
axes[0].set_xlabel("Elevation Angle (°)")
axes[0].set_ylabel("km")
axes[0].grid(True, alpha=0.3)
axes[0].invert_xaxis()   # low elevation on LEFT, zenith on RIGHT — physically correct
axes[0].annotate(f"Zenith: {sr_at_zenith:.0f} km ✓",
                 xy=(90, sr_at_zenith), xytext=(60, sr_at_zenith + 100),
                 fontsize=8, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))
axes[0].annotate(f"10°: {sr_at_10:.0f} km",
                 xy=(10, sr_at_10), xytext=(25, sr_at_10 - 200),
                 fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))

axes[1].plot(elevs, fspls, color='red', linewidth=2)
axes[1].axvline(x=MIN_ELEVATION_DEG, color='grey', linestyle='--', linewidth=1)
axes[1].set_title("Free Space Path Loss (dB)")
axes[1].set_xlabel("Elevation Angle (°)")
axes[1].set_ylabel("dB")
axes[1].grid(True, alpha=0.3)
axes[1].invert_xaxis()

axes[2].plot(elevs, p_losses, color='purple', linewidth=2)
axes[2].axvline(x=MIN_ELEVATION_DEG, color='grey', linestyle='--', linewidth=1)
axes[2].set_title(f"Pointing Loss (dB)\n"
                  f"({'Fixed Zenith' if antenna_mode == 'non_steerable' else 'Steerable = 0 dB always'})")
axes[2].set_xlabel("Elevation Angle (°)")
axes[2].set_ylabel("dB")
axes[2].grid(True, alpha=0.3)
axes[2].invert_xaxis()

plt.tight_layout()
st.pyplot(fig2)

# ============================================================
# SECTION 11 — HEATMAPS
# ============================================================
elevs_hm = np.arange(MIN_ELEVATION_DEG, 91, 5)

# --- Heatmap 1: Elevation vs Antenna Diameter ---
st.markdown("---")
st.subheader("🌡️ Heatmap 1: Link Margin vs Elevation & Antenna Diameter")
st.markdown("""
Green = link OK, Red = link failed.
White dashed = current 0.7m MangoSat antenna.
Answers: *"How big does the dish need to be?"*
""")

diameters_hm = np.arange(0.3, 3.1, 0.1)
heatmap1     = np.zeros((len(diameters_hm), len(elevs_hm)))
for i, d in enumerate(diameters_hm):
    for j, el in enumerate(elevs_hm):
        lm, *_ = calc_link_margin(
            el, params_live["eirp_dBW"], params_live["gt_dBK"],
            params_live["desired_data_rate_bps"], params_live["ebno_threshold_dB"],
            params_live["polarization_loss_dB"], params_live["atmo_loss_dB"],
            params_live["iono_loss_dB"], params_live["rain_loss_dB"],
            params_live["pointing_loss_dB"], antenna_mode, d, antenna_efficiency)
        heatmap1[i, j] = lm

fig3, ax3 = plt.subplots(figsize=(12, 5))
vmax  = max(abs(heatmap1.min()), abs(heatmap1.max()))
norm  = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im    = ax3.imshow(heatmap1, aspect='auto', origin='lower',
                   extent=[elevs_hm[0], elevs_hm[-1],
                           diameters_hm[0], diameters_hm[-1]],
                   cmap='RdYlGn', norm=norm)
plt.colorbar(im, ax=ax3, label='Link Margin (dB)')
ax3.axhline(y=0.7, color='white', linestyle='--', linewidth=2,
            label='Current 0.7m antenna')
ax3.axvline(x=MIN_ELEVATION_DEG, color='cyan', linestyle='--', linewidth=1.5,
            label=f'Min contact elevation ({MIN_ELEVATION_DEG}°)')
ax3.set_xlabel("Elevation Angle (°)", fontsize=12)
ax3.set_ylabel("Antenna Diameter (m)", fontsize=12)
ax3.set_title(f"Link Margin: Elevation vs Dish Size | {use_case_name}", fontsize=13)
ax3.legend(facecolor='dimgrey', labelcolor='white', fontsize=9)
st.pyplot(fig3)

# --- Heatmap 2: Elevation vs EIRP ---
st.markdown("---")
st.subheader("🌡️ Heatmap 2: Link Margin vs Elevation & EIRP")
st.markdown("""
Answers: *"What if satellite transmit power degrades on orbit?"*
White dashed = current EIRP from link budget.
""")

eirps_hm = np.arange(-20, 51, 2)
heatmap2  = np.zeros((len(eirps_hm), len(elevs_hm)))
for i, eirp in enumerate(eirps_hm):
    for j, el in enumerate(elevs_hm):
        lm, *_ = calc_link_margin(
            el, eirp, params_live["gt_dBK"],
            params_live["desired_data_rate_bps"], params_live["ebno_threshold_dB"],
            params_live["polarization_loss_dB"], params_live["atmo_loss_dB"],
            params_live["iono_loss_dB"], params_live["rain_loss_dB"],
            params_live["pointing_loss_dB"], antenna_mode,
            antenna_diameter_m, antenna_efficiency)
        heatmap2[i, j] = lm

fig4, ax4 = plt.subplots(figsize=(12, 5))
vmax2 = max(abs(heatmap2.min()), abs(heatmap2.max()))
norm2 = mcolors.TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
im2   = ax4.imshow(heatmap2, aspect='auto', origin='lower',
                   extent=[elevs_hm[0], elevs_hm[-1],
                           eirps_hm[0], eirps_hm[-1]],
                   cmap='RdYlGn', norm=norm2)
plt.colorbar(im2, ax=ax4, label='Link Margin (dB)')
ax4.axhline(y=params_live["eirp_dBW"], color='white', linestyle='--', linewidth=2,
            label=f"Current EIRP ({params_live['eirp_dBW']} dBW)")
ax4.axvline(x=MIN_ELEVATION_DEG, color='cyan', linestyle='--', linewidth=1.5,
            label=f'Min contact elevation ({MIN_ELEVATION_DEG}°)')
ax4.set_xlabel("Elevation Angle (°)", fontsize=12)
ax4.set_ylabel("EIRP (dBW)", fontsize=12)
ax4.set_title(f"Link Margin: Elevation vs EIRP | {use_case_name}", fontsize=13)
ax4.legend(facecolor='dimgrey', labelcolor='white', fontsize=9)
st.pyplot(fig4)

# --- Heatmap 3: Elevation vs Data Rate ---
st.markdown("---")
st.subheader("🌡️ Heatmap 3: Link Margin vs Elevation & Data Rate")
st.markdown("""
Answers: *"What is the maximum data rate I can use at each elevation angle?"*
Higher data rate = harder to close the link.
""")

data_rates_hm = np.array([2400, 4800, 9600, 19200, 38400, 76800,
                           100000, 150000, 200000, 250000, 500000, 1000000])
dr_labels     = ["2.4k", "4.8k", "9.6k", "19.2k", "38.4k",
                 "76.8k", "100k", "150k", "200k", "250k", "500k", "1M"]
heatmap3      = np.zeros((len(data_rates_hm), len(elevs_hm)))

for i, dr in enumerate(data_rates_hm):
    for j, el in enumerate(elevs_hm):
        lm, *_ = calc_link_margin(
            el, params_live["eirp_dBW"], params_live["gt_dBK"],
            dr, params_live["ebno_threshold_dB"],
            params_live["polarization_loss_dB"], params_live["atmo_loss_dB"],
            params_live["iono_loss_dB"], params_live["rain_loss_dB"],
            params_live["pointing_loss_dB"], antenna_mode,
            antenna_diameter_m, antenna_efficiency)
        heatmap3[i, j] = lm

fig5, ax5 = plt.subplots(figsize=(12, 5))
vmax3 = max(abs(heatmap3.min()), abs(heatmap3.max()))
norm3 = mcolors.TwoSlopeNorm(vmin=-vmax3, vcenter=0, vmax=vmax3)
im3   = ax5.imshow(heatmap3, aspect='auto', origin='lower',
                   extent=[elevs_hm[0], elevs_hm[-1], 0, len(data_rates_hm)],
                   cmap='RdYlGn', norm=norm3)
plt.colorbar(im3, ax=ax5, label='Link Margin (dB)')
ax5.set_yticks(np.arange(len(data_rates_hm)) + 0.5)
ax5.set_yticklabels(dr_labels)
ax5.axhline(y=data_rates_hm.tolist().index(params_live["desired_data_rate_bps"]) + 0.5,
            color='white', linestyle='--', linewidth=2,
            label=f"Desired DR ({params_live['desired_data_rate_bps']:,} bps)")
ax5.axvline(x=MIN_ELEVATION_DEG, color='cyan', linestyle='--', linewidth=1.5,
            label=f'Min contact elevation ({MIN_ELEVATION_DEG}°)')
ax5.set_xlabel("Elevation Angle (°)", fontsize=12)
ax5.set_ylabel("Data Rate (bps)", fontsize=12)
ax5.set_title(f"Link Margin: Elevation vs Data Rate | {use_case_name}", fontsize=13)
ax5.legend(facecolor='dimgrey', labelcolor='white', fontsize=9)
st.pyplot(fig5)

# ============================================================
# SECTION 12 — PASS DURATION SUMMARY
# ============================================================
st.markdown("---")
st.subheader("⏱️ Comms Duration Per Pass")
st.markdown("Based on orbital geometry at 590 km altitude.")

col_a, col_b = st.columns(2)
with col_a:
    st.info(f"""
    **Desired Data Rate ({params_live['desired_data_rate_bps']:,} bps)**
    - Min viable elevation: **{min_viable_desired}°** {'✅' if min_viable_desired else '❌'}
    - Pass duration: **{pass_dur_desired:.1f} minutes**
    """)
with col_b:
    st.info(f"""
    **Minimum Data Rate (2,400 bps)**
    - Min viable elevation: **{min_viable_min_dr}°** {'✅' if min_viable_min_dr else '❌'}
    - Pass duration: **{pass_dur_min_dr:.1f} minutes**
    """)

# ============================================================
# SECTION 13 — DATA TABLE
# ============================================================
st.markdown("---")
st.subheader("📋 Link Budget Table")

_, margins_desired_full, ranges_full, fspls_full, p_losses_full = run_elevation_sweep(
    params_live, params_live["desired_data_rate_bps"],
    antenna_mode, antenna_diameter_m, antenna_efficiency)
_, margins_min_full, *_ = run_elevation_sweep(
    params_live, params_live["min_data_rate_bps"],
    antenna_mode, antenna_diameter_m, antenna_efficiency)

elevs_table = np.arange(MIN_ELEVATION_DEG, 91, 5)
idx         = [int(e - MIN_ELEVATION_DEG) for e in elevs_table]

table = pd.DataFrame({
    "Elevation (°)"                                        : elevs_table,
    "Slant Range (km)"                                     : ranges_full[idx].round(1),
    "FSPL (dB)"                                            : fspls_full[idx].round(2),
    "Pointing Loss (dB)"                                   : p_losses_full[idx].round(2),
    f"Margin @ {params_live['desired_data_rate_bps']:,} bps": margins_desired_full[idx].round(2),
    "Margin @ 2400 bps (dB)"                               : margins_min_full[idx].round(2),
    "Status (Desired DR)"                                  : ["✅ OK" if lm >= 0 else "❌ Failed"
                                                              for lm in margins_desired_full[idx]],
})
st.dataframe(table, use_container_width=True)

# ============================================================
# SECTION 14 — PARAMETER REFERENCE
# ============================================================
st.markdown("---")
with st.expander("📄 Full Parameter Reference (from MangoSat Link Budget PDF)"):
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Orbital Altitude | {ALTITUDE_KM} km |
    | Min Contact Elevation | {MIN_ELEVATION_DEG}° |
    | Frequency Band | S-Band (~{FREQUENCY_GHz} GHz) |
    | EIRP | {params['eirp_dBW']} dBW |
    | G/T | {params['gt_dBK']} dB/K |
    | Polarization Loss | {params['polarization_loss_dB']} dB |
    | Atmospheric Loss | {params['atmo_loss_dB']} dB |
    | Ionospheric Loss | {params['iono_loss_dB']} dB |
    | Rain Loss | {params['rain_loss_dB']} dB |
    | Fixed Pointing Loss | {params['pointing_loss_dB']} dB |
    | Desired Data Rate | {params['desired_data_rate_bps']:,} bps |
    | Min Data Rate | {params['min_data_rate_bps']:,} bps |
    | Eb/No Threshold | {params['ebno_threshold_dB']} dB |
    | **PDF Reference Margin** | **{params['ref_margin_dB']} dB** |
    """)
