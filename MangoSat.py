# ============================================================
# LINK MARGIN CALCULATOR — Zenith Fixed Antenna
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
SPEED_OF_LIGHT = 3e8
EARTH_RADIUS_KM = 6371
BOLTZMANN_dBWKHz = -228.6

# ============================================================
# FUNCTIONS
# ============================================================

def calc_slant_range(elevation_deg, altitude_km):
    El = np.radians(elevation_deg)
    Re = EARTH_RADIUS_KM
    h = altitude_km
    slant_range_km = np.sqrt((Re * np.cos(El))**2 + 2*Re*h + h**2) - Re * np.cos(El)
    return slant_range_km

def calc_fspl_dB(slant_range_km, frequency_GHz):
    d = slant_range_km * 1000
    f = frequency_GHz * 1e9
    fspl = 20 * np.log10((4 * np.pi * d * f) / SPEED_OF_LIGHT)
    return fspl

def calc_antenna_gain_dB(diameter_m, frequency_GHz, efficiency):
    f = frequency_GHz * 1e9
    wavelength = SPEED_OF_LIGHT / f
    gain_linear = efficiency * (np.pi * diameter_m / wavelength)**2
    gain_dB = 10 * np.log10(gain_linear)
    return gain_dB

def calc_beamwidth_deg(diameter_m, frequency_GHz):
    f = frequency_GHz * 1e9
    wavelength = SPEED_OF_LIGHT / f
    beamwidth = np.degrees(1.22 * wavelength / diameter_m)
    return beamwidth

def calc_pointing_loss_dB(elevation_deg, beamwidth_deg):
    theta = 90 - elevation_deg
    loss = -12 * (theta / beamwidth_deg)**2
    return loss

def calc_GT_dB(antenna_gain_dB, system_noise_temp_K):
    G_T = antenna_gain_dB - 10 * np.log10(system_noise_temp_K)
    return G_T

def calc_link_margin(elevation_deg, EIRP_dBW, altitude_km, diameter_m,
                     efficiency, frequency_GHz, noise_temp_K,
                     required_EbNo_dB, data_rate_kbps):
    slant_range_km = calc_slant_range(elevation_deg, altitude_km)
    fspl_dB = calc_fspl_dB(slant_range_km, frequency_GHz)
    antenna_gain_dB = calc_antenna_gain_dB(diameter_m, frequency_GHz, efficiency)
    beamwidth_deg = calc_beamwidth_deg(diameter_m, frequency_GHz)
    pointing_loss_dB = calc_pointing_loss_dB(elevation_deg, beamwidth_deg)
    GT_dB = calc_GT_dB(antenna_gain_dB, noise_temp_K)
    data_rate_Hz = data_rate_kbps * 1000
    data_rate_dBHz = 10 * np.log10(data_rate_Hz)
    link_margin = (EIRP_dBW + GT_dB + pointing_loss_dB
                   - fspl_dB - BOLTZMANN_dBWKHz
                   - data_rate_dBHz - required_EbNo_dB)
    return link_margin, slant_range_km, fspl_dB, antenna_gain_dB, pointing_loss_dB, GT_dB

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Link Margin Calculator", layout="wide")
st.title("📡 Link Margin vs Elevation Angle")
st.markdown("""
> **What is this app?**  
> When a satellite passes overhead, your ground antenna (fixed pointing straight up — called **zenith**) 
> cannot follow it. As the satellite moves away from directly above, the signal weakens.  
> This app shows **how much signal margin you have** at every elevation angle — 
> and at what angle the link breaks down.
""")

# ============================================================
# SIDEBAR — Input Mode Toggle
# ============================================================
st.sidebar.header("⚙️ System Parameters")
input_mode = st.sidebar.radio("Input Mode", ["🎚️ Sliders", "⌨️ Manual Entry"])

st.sidebar.markdown("---")

def get_input(label, min_val, max_val, default, step, key, input_mode, format="%.2f"):
    if input_mode == "🎚️ Sliders":
        if isinstance(default, float):
            return st.sidebar.slider(label, float(min_val), float(max_val), float(default), float(step), key=key)
        else:
            return st.sidebar.slider(label, int(min_val), int(max_val), int(default), int(step), key=key)
    else:
        return st.sidebar.number_input(label, min_value=float(min_val), max_value=float(max_val),
                                       value=float(default), step=float(step), key=key, format=format)

# ============================================================
# SIDEBAR — Parameters
# ============================================================
st.sidebar.subheader("🛰️ Satellite")
EIRP_dBW = get_input("Satellite EIRP (dBW)", 20, 60, 40, 1, "eirp", input_mode, "%.0f")
satellite_altitude_km = get_input("Satellite Altitude (km)", 300, 1200, 600, 50, "alt", input_mode, "%.0f")

st.sidebar.subheader("📡 Ground Antenna")
antenna_diameter_m = get_input("Antenna Diameter (m)", 0.5, 5.0, 1.2, 0.1, "diam", input_mode)
antenna_efficiency = get_input("Antenna Efficiency (η)", 0.4, 0.9, 0.6, 0.05, "eff", input_mode)
frequency_GHz = get_input("Frequency (GHz)", 1.0, 30.0, 8.0, 0.5, "freq", input_mode)
system_noise_temp_K = get_input("System Noise Temperature (K)", 50, 500, 150, 10, "noise", input_mode, "%.0f")

st.sidebar.subheader("🔗 Link")
required_EbNo_dB = get_input("Required Eb/No (dB)", 3, 20, 10, 1, "ebno", input_mode, "%.0f")
data_rate_kbps = get_input("Data Rate (kbps)", 1, 1000, 100, 10, "dr", input_mode, "%.0f")

# ============================================================
# CALCULATIONS
# ============================================================
elevation_angles = np.arange(5, 91, 1)
link_margins, slant_ranges, fspl_values, pointing_losses = [], [], [], []

for el in elevation_angles:
    lm, sr, fspl, ag, pl, gt = calc_link_margin(
        el, EIRP_dBW, satellite_altitude_km,
        antenna_diameter_m, antenna_efficiency,
        frequency_GHz, system_noise_temp_K,
        required_EbNo_dB, data_rate_kbps)
    link_margins.append(lm)
    slant_ranges.append(sr)
    fspl_values.append(fspl)
    pointing_losses.append(pl)

link_margins  = np.array(link_margins)
slant_ranges  = np.array(slant_ranges)
fspl_values   = np.array(fspl_values)
pointing_losses = np.array(pointing_losses)

beamwidth = calc_beamwidth_deg(antenna_diameter_m, frequency_GHz)
lm_zenith = link_margins[-1]
lm_low    = link_margins[0]
min_viable_el = next((elevation_angles[i] for i, lm in enumerate(link_margins) if lm >= 0), None)

# ============================================================
# KEY METRICS
# ============================================================
st.markdown("---")
st.subheader("📊 Key Metrics")
st.markdown("""
These four numbers summarise your link at a glance.  
- **Link Margin at Zenith** — best-case signal (satellite directly overhead).  
- **Link Margin at 5°** — worst-case signal (satellite near horizon).  
- **Min Viable Elevation** — the lowest angle where the link still works (margin ≥ 0 dB).  
- **Antenna Beamwidth** — the cone of sky your antenna can "see" well. Narrow beam = more pointing loss at low elevation.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Link Margin at Zenith (90°)", f"{lm_zenith:.1f} dB",
              delta="Best case ✅" if lm_zenith > 0 else "Failed ❌")
with col2:
    st.metric("Link Margin at 5°", f"{lm_low:.1f} dB",
              delta="OK ✅" if lm_low > 0 else "Failed ❌")
with col3:
    if min_viable_el:
        st.metric("Min Viable Elevation", f"{min_viable_el}°", delta="Link works above this ✅")
    else:
        st.metric("Min Viable Elevation", "None", delta="Link always fails ❌")
with col4:
    st.metric("Antenna Beamwidth", f"{beamwidth:.1f}°")

# ============================================================
# MAIN PLOT
# ============================================================
st.markdown("---")
st.subheader("📈 Link Margin vs Elevation Angle")
st.markdown("""
**How to read this chart:**  
The blue line is your link margin at each elevation angle.  
🟢 Green zone = link is working (margin above 0 dB).  
🔴 Red zone = link has failed (margin below 0 dB).  
The dashed red line at 0 dB is the pass/fail threshold.
""")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(elevation_angles, link_margins, color='royalblue', linewidth=2.5, label='Link Margin')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='0 dB threshold')
ax.fill_between(elevation_angles, link_margins, 0,
                where=(link_margins >= 0), alpha=0.15, color='green', label='Link OK')
ax.fill_between(elevation_angles, link_margins, 0,
                where=(link_margins < 0), alpha=0.15, color='red', label='Link Failed')
if min_viable_el:
    ax.axvline(x=min_viable_el, color='orange', linestyle=':', linewidth=2,
               label=f'Min viable elevation ({min_viable_el}°)')
ax.set_xlabel("Elevation Angle (degrees)", fontsize=12)
ax.set_ylabel("Link Margin (dB)", fontsize=12)
ax.set_title("Link Margin vs Elevation Angle (Fixed Zenith Antenna)", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([5, 90])
st.pyplot(fig)

# ============================================================
# LOSS BREAKDOWN PLOTS
# ============================================================
st.markdown("---")
st.subheader("🔍 Loss Breakdown vs Elevation Angle")
st.markdown("""
These three charts show **why** link margin drops at low elevation angles — broken into individual contributors:

- **Slant Range** — the actual distance the signal travels. At low elevation the satellite is further away (geometry), so signal has to travel further.
- **Free Space Path Loss (FSPL)** — signal naturally spreads out over distance. More distance = more loss. This is physics, unavoidable.
- **Pointing Loss** — your antenna is fixed pointing up. As the satellite moves away from zenith, your antenna beam misses it. This loss grows rapidly at low elevation.
""")

fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(elevation_angles, slant_ranges, color='orange', linewidth=2)
axes[0].set_title("Slant Range (km)")
axes[0].set_xlabel("Elevation Angle (°)")
axes[0].set_ylabel("km")
axes[0].grid(True, alpha=0.3)

axes[1].plot(elevation_angles, fspl_values, color='red', linewidth=2)
axes[1].set_title("Free Space Path Loss (dB)")
axes[1].set_xlabel("Elevation Angle (°)")
axes[1].set_ylabel("dB")
axes[1].grid(True, alpha=0.3)

axes[2].plot(elevation_angles, pointing_losses, color='purple', linewidth=2)
axes[2].set_title("Pointing Loss (dB)")
axes[2].set_xlabel("Elevation Angle (°)")
axes[2].set_ylabel("dB")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)

# ============================================================
# HEATMAP 1 — Link Margin vs Elevation & Altitude
# ============================================================
st.markdown("---")
st.subheader("🌡️ Heatmap 1: Link Margin vs Elevation Angle & Satellite Altitude")
st.markdown("""
**How to read this heatmap:**  
Each cell shows the link margin (dB) for a combination of elevation angle (X axis) and satellite altitude (Y axis).  
🟢 Green = strong link. 🔴 Red = link failure.  
This tells you: *"If my satellite is at X altitude and Y elevation, is my link OK?"*  
Notice how higher altitude = worse margin (longer path), and low elevation = worse margin (pointing loss + longer path).
""")

altitudes_hm   = np.arange(300, 1250, 50)
elevations_hm  = np.arange(5, 91, 5)
heatmap1 = np.zeros((len(altitudes_hm), len(elevations_hm)))

for i, alt in enumerate(altitudes_hm):
    for j, el in enumerate(elevations_hm):
        lm, *_ = calc_link_margin(el, EIRP_dBW, alt, antenna_diameter_m,
                                   antenna_efficiency, frequency_GHz,
                                   system_noise_temp_K, required_EbNo_dB, data_rate_kbps)
        heatmap1[i, j] = lm

fig3, ax3 = plt.subplots(figsize=(12, 6))
norm = mcolors.TwoSlopeNorm(vmin=heatmap1.min(), vcenter=0, vmax=heatmap1.max())
im = ax3.imshow(heatmap1, aspect='auto', origin='lower',
                extent=[elevations_hm[0], elevations_hm[-1],
                        altitudes_hm[0], altitudes_hm[-1]],
                cmap='RdYlGn', norm=norm)
plt.colorbar(im, ax=ax3, label='Link Margin (dB)')
ax3.set_xlabel("Elevation Angle (°)", fontsize=12)
ax3.set_ylabel("Satellite Altitude (km)", fontsize=12)
ax3.set_title("Link Margin Heatmap: Elevation vs Altitude", fontsize=14)
ax3.axvline(x=min_viable_el, color='white', linestyle='--', linewidth=2,
            label=f'Current min viable elevation ({min_viable_el}°)') if min_viable_el else None
ax3.legend(facecolor='grey')
st.pyplot(fig3)

# ============================================================
# HEATMAP 2 — Link Margin vs Elevation & Antenna Diameter
# ============================================================
st.markdown("---")
st.subheader("🌡️ Heatmap 2: Link Margin vs Elevation Angle & Antenna Diameter")
st.markdown("""
**How to read this heatmap:**  
X axis = elevation angle. Y axis = antenna dish diameter.  
This answers: *"How big does my dish need to be to maintain a link at low elevation angles?"*  
A bigger dish gives more gain BUT also a narrower beam — so pointing loss gets worse faster.  
Watch for the sweet spot where the dish is large enough to collect signal but beam isn't too narrow.
""")

diameters_hm  = np.arange(0.5, 5.1, 0.25)
heatmap2 = np.zeros((len(diameters_hm), len(elevations_hm)))

for i, diam in enumerate(diameters_hm):
    for j, el in enumerate(elevations_hm):
        lm, *_ = calc_link_margin(el, EIRP_dBW, satellite_altitude_km, diam,
                                   antenna_efficiency, frequency_GHz,
                                   system_noise_temp_K, required_EbNo_dB, data_rate_kbps)
        heatmap2[i, j] = lm

fig4, ax4 = plt.subplots(figsize=(12, 6))
norm2 = mcolors.TwoSlopeNorm(vmin=heatmap2.min(), vcenter=0, vmax=heatmap2.max())
im2 = ax4.imshow(heatmap2, aspect='auto', origin='lower',
                 extent=[elevations_hm[0], elevations_hm[-1],
                         diameters_hm[0], diameters_hm[-1]],
                 cmap='RdYlGn', norm=norm2)
plt.colorbar(im2, ax=ax4, label='Link Margin (dB)')
ax4.set_xlabel("Elevation Angle (°)", fontsize=12)
ax4.set_ylabel("Antenna Diameter (m)", fontsize=12)
ax4.set_title("Link Margin Heatmap: Elevation vs Antenna Diameter", fontsize=14)
st.pyplot(fig4)

# ============================================================
# HEATMAP 3 — Link Margin vs Elevation & Frequency
# ============================================================
st.markdown("---")
st.subheader("🌡️ Heatmap 3: Link Margin vs Elevation Angle & Frequency")
st.markdown("""
**How to read this heatmap:**  
X axis = elevation angle. Y axis = frequency band.  
Higher frequency = more path loss BUT also more antenna gain for the same dish size.  
This heatmap shows which frequency band performs best for your dish size at each elevation angle.  
Common bands for reference: L-band ~1.5 GHz, S-band ~2.5 GHz, X-band ~8 GHz, Ka-band ~26 GHz.
""")

frequencies_hm = np.arange(1.0, 30.5, 1.0)
heatmap3 = np.zeros((len(frequencies_hm), len(elevations_hm)))

for i, freq in enumerate(frequencies_hm):
    for j, el in enumerate(elevations_hm):
        lm, *_ = calc_link_margin(el, EIRP_dBW, satellite_altitude_km,
                                   antenna_diameter_m, antenna_efficiency,
                                   freq, system_noise_temp_K,
                                   required_EbNo_dB, data_rate_kbps)
        heatmap3[i, j] = lm

fig5, ax5 = plt.subplots(figsize=(12, 6))
norm3 = mcolors.TwoSlopeNorm(vmin=heatmap3.min(), vcenter=0, vmax=heatmap3.max())
im3 = ax5.imshow(heatmap3, aspect='auto', origin='lower',
                 extent=[elevations_hm[0], elevations_hm[-1],
                         frequencies_hm[0], frequencies_hm[-1]],
                 cmap='RdYlGn', norm=norm3)
plt.colorbar(im3, ax=ax5, label='Link Margin (dB)')
ax5.set_xlabel("Elevation Angle (°)", fontsize=12)
ax5.set_ylabel("Frequency (GHz)", fontsize=12)
ax5.set_title("Link Margin Heatmap: Elevation vs Frequency", fontsize=14)
ax5.axhline(y=frequency_GHz, color='white', linestyle='--', linewidth=2,
            label=f'Current frequency ({frequency_GHz} GHz)')
ax5.legend(facecolor='grey')
st.pyplot(fig5)

# ============================================================
# DATA TABLE
# ============================================================
st.markdown("---")
st.subheader("📋 Data Table")
st.markdown("""
Raw numbers at every 10° elevation angle. Useful for reporting or quick sanity checks.  
You can hover over column headers to sort.
""")

table_data = pd.DataFrame({
    "Elevation (°)"      : elevation_angles[::10],
    "Slant Range (km)"   : slant_ranges[::10].round(1),
    "FSPL (dB)"          : fspl_values[::10].round(2),
    "Pointing Loss (dB)" : pointing_losses[::10].round(2),
    "Link Margin (dB)"   : link_margins[::10].round(2),
    "Status"             : ["✅ OK" if lm >= 0 else "❌ Failed" for lm in link_margins[::10]]
})
st.dataframe(table_data, use_container_width=True)

# ============================================================
# FOOTER EXPLANATION
# ============================================================
st.markdown("---")
with st.expander("📖 Click here — Full explanation of every parameter & equation used"):
    st.markdown("""
    ### Parameters Explained

    | Parameter | Unit | What it means |
    |---|---|---|
    | **EIRP** | dBW | Effective Isotropic Radiated Power — how loudly the satellite transmits |
    | **Satellite Altitude** | km | Orbital height. LEO = 300–1200 km |
    | **Antenna Diameter** | m | Physical size of your dish. Bigger = more gain, narrower beam |
    | **Antenna Efficiency (η)** | — | No dish is perfect. 0.6 = 60% efficient. Accounts for surface errors, feed losses |
    | **Frequency** | GHz | Signal frequency. Higher = more path loss, more antenna gain |
    | **System Noise Temperature** | K | How noisy your receiver is. Lower = better. Cryogenic receivers go to ~30K |
    | **Required Eb/No** | dB | Minimum signal-to-noise per bit needed for your modulation scheme |
    | **Data Rate** | kbps | Higher data rate = harder to close the link |

    ### Equations Used

    **Slant Range:**
                R = sqrt((Re·cos(El))² + 2·Re·h + h²) - Re·cos(El)
                **Free Space Path Loss:**
                FSPL = 20·log10(4π·d·f / c)
                **Antenna Gain:**
                G = η · (π·D·f/c)²   → converted to dB

    **Beamwidth:**
θ_3dB = 1.22 · λ / D   (radians, converted to degrees)

    **Pointing Loss (fixed zenith antenna):**
L_point = -12 · (θ_offset / θ_3dB)²
where θ_offset = 90° - elevation_angle

    **G/T (Figure of Merit):**
G/T = G_rx - 10·log10(T_sys)

    **Link Margin:**
LM = EIRP + G/T + L_point - FSPL - k - R_b - (Eb/No)_req
where k = Boltzmann constant = -228.6 dBW/K/Hz
    """)
