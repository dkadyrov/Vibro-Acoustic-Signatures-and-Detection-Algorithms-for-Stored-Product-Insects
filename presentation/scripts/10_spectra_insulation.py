# %%
from datetime import timedelta

import cblind as cb  # type: ignore  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dankpy import color  # type: ignore
from scipy import signal
from spidb import spidb

from aspids_tools import normalization

plt.style.use("dankpy.styles.stevens_presentation")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")

record = db.session.get(spidb.Record, 464)

reference = normalization.reference_signal(db, record.sensor, channels=[4, 5])

channels = [0, 1, 4, 5, 7]

audios = db.get_audios(
    record.sensor,
    start=record.start + timedelta(seconds=40),  # + timedelta(seconds=51.40),
    end=record.end,  # record.start + timedelta(seconds=51.50),
    channels=channels,
)

for i, audio in enumerate(audios):
    audios[i] = audio.trim(start=11.40, end=11.50, time_format="seconds", restart=False)
# audios = db.get_audios
#     record.sensor,
#     start=record.start + timedelta(seconds=50),
#     end=record.start + timedelta(seconds=60),
#     channels=channels,
# )

labels = [
    "Ch. 0 (Piezoelectric)",
    "Ch. 1 (Piezoelectric)",
    "Ch. 4 (Internal Microphone)",
    "Ch. 5 (Internal Microphone)",
    "Ch. 7 (External Microphone)",
]

z_int_pie = [-130, -90]
z_int_mic = [-125, -95]
z_ext_mic = [-90, -50]

# for a in audios:
# a.data = a.data.dropna().reset_index(drop=True) # Commented out to prevent whitespace in spectrogram
# pass

fig, axs = plt.subplots(nrows=len(audios), ncols=1, sharex=True, figsize=(11.71, 4.24))

# Store image objects for colorbars
img_objects = {}

for i, a in enumerate(audios):
    ax = axs[i]

    times, frequencies, spectrogram = a.spectrogram(
        window="hann",
        window_size=128,
        nperseg=128,
        nfft=128,
        noverlap=64,  # Increased overlap to reduce whitespace
        time_format="seconds",
    )

    spectrogram = 10 * np.log10(np.abs(spectrogram))

    # Calculate proper extents to eliminate whitespace
    # Use actual time duration of the signal
    signal_duration = len(a.data.signal) / a.sample_rate
    extents = [
        0,  # Start at 0
        signal_duration,  # End at actual signal duration
        frequencies.min(),
        frequencies.max(),
    ]
    axi = ax.imshow(
        spectrogram,
        extent=extents,
        # cmap="jet",
        # cmap="cividis_r",
        # cmap="viridis_r",
        cmap="cb.solstice",
        aspect="auto",
        origin="lower",
    )

    if i in [0, 1]:
        zmin, zmax = -130, -90
        img_objects["internal"] = axi
    elif i in [2, 3]:
        zmin, zmax = -125, -95
        img_objects["external"] = axi
    else:
        zmin, zmax = -90, -50
        img_objects["last"] = axi

    axi.set_clim([zmin, zmax])
    # add a blank line to the plot to add a label to the legend
    ax.plot([], [], "", label=f"{labels[i]}")  # this wasn't blank
    ax.legend(
        loc="upper right",
        handlelength=0,
        handletextpad=0,
    )

    # change the x-axis tick labels to be 11.4 to 11.5
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, signal_duration])
    ax.set_xticklabels(
        [
            f"{11.4:.1f}",
            f"{11.42:.2f}",
            f"{11.44:.2f}",
            f"{11.46:.2f}",
            f"{11.48:.2f}",
            f"{11.5:.1f}",
        ]
    )

    # Set x-axis to cover the full signal duration without whitespace
    signal_duration = len(a.data.signal) / a.sample_rate
    ax.set_xlim([0, signal_duration])
    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 4000, 8000])

fig.supxlabel("Time [s]", fontsize=10)
fig.supylabel("Frequency [Hz]", fontsize=10)

# Create separate colorbars for each group
# Colorbar for internal mics (channels 0, 1)
cbar1 = fig.colorbar(
    img_objects["internal"],
    ax=[axs[0], axs[1]],
    orientation="vertical",
    location="right",
    aspect=25,
    ticks=[-130, -110, -90],
    pad=0.02,
)

# Colorbar for external mics (channels 2, 3)
cbar2 = fig.colorbar(
    img_objects["external"],
    ax=[axs[2], axs[3]],
    orientation="vertical",
    location="right",
    aspect=25,
    ticks=[-125, -110, -95],
    pad=0.02,
)
cbar2.ax.set_ylabel("Power [dB]", fontsize=10)

# Colorbar for last external mic (channel 4)
cbar3 = fig.colorbar(
    img_objects["last"],
    ax=axs[4],
    orientation="vertical",
    location="right",
    aspect=12.5,
    ticks=[-90, -70, -50],
    pad=0.02,
)
# fig.savefig(r"projects\MDPI-Detection\paper\figures\noise_impulse_zoom.pdf", dpi=300)
# %%
fig, ax = plt.subplots()
psd_data = {}
frequencies = None

# labels = ["Ch. 4 (Internal Mic.)", "Ch. 5 (Internal Mic.)", "Ch. 7 (External Mic.)"]

i = 0
for a, audio in enumerate(audios):
    if a in [0, 1]:
        continue
    f, p = signal.welch(
        audio.data.signal.values,
        fs=audio.sample_rate,
        nperseg=512,
        nfft=512,
        noverlap=256,
        window="blackmanharris",
        average="mean",
        scaling="spectrum",
    )
    p = 10 * np.log10(p)

    # Store the data
    psd_data[audio.channel_number] = p
    if frequencies is None:
        frequencies = f

    ax.plot(
        f,
        p,
        label=f"Ch. {channels[a]}",
        color=color.colors[i],
    )
    i += 1
ax.plot(
    reference["frequency"],
    reference["average"],
    label="Reference",
    color="black",
    linestyle="-",
)

ax.set_xlim(0, 8000)
ax.legend(loc="upper right", ncols=4)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
ax.set_ylim(-120, 0)
ax.set_yticks([-120, -60, 0])
ax.set_title("Noise 90 dBA")
# fig.savefig(
#     r"projects\Dissertation\dissertation\figures\5_2_noise_impulse_zoom_spectra.pdf",
#     dpi=300,
# )
# %%
# Calculate differences between Ch. 7 and Ch. 5
if 7 in psd_data and 5 in psd_data:
    ch7_psd = psd_data[7]
    ch5_psd = psd_data[5]

    # Calculate difference (Ch. 7 - Ch. 5)
    difference = ch7_psd - ch5_psd

    # Find maximum difference
    max_difference = np.max(difference)
    max_diff_freq = frequencies[np.argmax(difference)]

    # Find average difference above 2000 Hz
    freq_mask = frequencies > 2000
    avg_difference_above_2khz = np.mean(difference[freq_mask])

    # Find maximum difference above 2000 Hz
    max_difference_above_2khz = np.max(difference[freq_mask])
    max_diff_freq_above_2khz = frequencies[freq_mask][np.argmax(difference[freq_mask])]

    print(
        f"Maximum difference between Ch. 7 and Ch. 5: {max_difference:.2f} dB at {max_diff_freq:.1f} Hz"
    )
    print(f"Average difference above 2000 Hz: {avg_difference_above_2khz:.2f} dB")
    print(
        f"Maximum difference above 2000 Hz: {max_difference_above_2khz:.2f} dB at {max_diff_freq_above_2khz:.1f} Hz"
    )
else:
    print("Error: Ch. 5 or Ch. 7 data not found")


# %%
# Calculate differences between Ch. 7 and Ch. 4
if 7 in psd_data and 4 in psd_data:
    ch7_psd = psd_data[7]
    ch4_psd = psd_data[4]

    # Calculate difference (Ch. 7 - Ch. 4)
    difference = ch7_psd - ch4_psd

    # Find maximum difference
    max_difference = np.max(difference)
    max_diff_freq = frequencies[np.argmax(difference)]

    # Find average difference above 2000 Hz
    freq_mask = frequencies > 2000
    avg_difference_above_2khz = np.mean(difference[freq_mask])

    # Find maximum difference above 2000 Hz
    max_difference_above_2khz = np.max(difference[freq_mask])
    max_diff_freq_above_2khz = frequencies[freq_mask][np.argmax(difference[freq_mask])]

    print(
        f"Maximum difference between Ch. 7 and Ch. 4: {max_difference:.2f} dB at {max_diff_freq:.1f} Hz"
    )
    print(f"Average difference above 2000 Hz: {avg_difference_above_2khz:.2f} dB")
    print(
        f"Maximum difference above 2000 Hz: {max_difference_above_2khz:.2f} dB at {max_diff_freq_above_2khz:.1f} Hz"
    )
else:
    print("Error: Ch. 5 or Ch. 7 data not found")
# %%
fig, ax = plt.subplots()
psd_data = {}
frequencies = None

# labels = ["Ch. 4 (Internal Mic.)", "Ch. 5 (Internal Mic.)", "Ch. 7 (External Mic.)"]

i = 0
for a, audio in enumerate(audios):
    if a in [0, 2, 4]:
        f, p = signal.welch(
            audio.data.signal.values,
            fs=audio.sample_rate,
            nperseg=512,
            nfft=512,
            noverlap=256,
            window="blackmanharris",
            average="mean",
            scaling="spectrum",
        )
        p = 10 * np.log10(p)

        # Store the data
        psd_data[audio.channel_number] = p
        if frequencies is None:
            frequencies = f

        ax.plot(
            f,
            p,
            label=f"Ch. {channels[a]}",
            color=color.colors[i],
        )
        i += 1
ax.plot(
    reference["frequency"],
    reference["average"],
    label="Reference",
    color="black",
    linestyle="-",
)

ax.set_xlim(0, 8000)
ax.legend(loc="upper right", ncols=4)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
ax.set_ylim(-120, 0)
ax.set_yticks([-120, -60, 0])
# fig.savefig(
#     r"projects\Dissertation\dissertation\figures\5_4_noise_impulse_zoom_spectra_piezo.pdf",
#     dpi=300,
# )
# %%
