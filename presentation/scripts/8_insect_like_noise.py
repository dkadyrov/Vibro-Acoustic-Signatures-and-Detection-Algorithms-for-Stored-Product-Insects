# %%
from datetime import timedelta

import cblind as cb  # type: ignore  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    start=record.start + timedelta(seconds=40),
    end=record.end,  # + timedelta(seconds=51.50),
    channels=channels,
)

labels = [
    "Ch. 0",
    "Ch. 1",
    "Ch. 4",
    "Ch. 5",
    "Ch. 7",
]

z_int_pie = [-130, -90]
z_int_mic = [-140, -100]
z_ext_mic = [-90, -50]

fig, axs = plt.subplots(nrows=len(audios), ncols=1, sharex=True, figsize=(11.71, 5))

# Store image objects for colorbars
img_objects = {}

for i, a in enumerate(audios):
    ax = axs[i]

    times, frequencies, spectrogram = a.spectrogram(
        window="hann",
        window_size=128,
        nperseg=128,
        nfft=128,
        noverlap=120,  # Increased overlap to reduce whitespace
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
        zmin, zmax = z_int_pie
        img_objects["internal"] = axi
    elif i in [2, 3]:
        zmin, zmax = z_int_mic
        img_objects["external"] = axi
    else:
        zmin, zmax = z_ext_mic
        img_objects["last"] = axi

    axi.set_clim([zmin, zmax])
    # add a blank line to the plot to add a label to the legend
    ax.plot([], [], "", label=f"{labels[i]}")  # this wasn't blank
    ax.legend(
        loc="upper right",
        handlelength=0,
        handletextpad=0,
    )

    # Set x-axis to cover the full signal duration without whitespace
    signal_duration = len(a.data.signal) / a.sample_rate
    ax.set_xlim([0, signal_duration])
    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 4000, 8000])

fig.supxlabel("Time [s]")
fig.supylabel("Frequency [Hz]")

# Create separate colorbars for each group
# Colorbar for internal mics (channels 0, 1)
cbar1 = fig.colorbar(
    img_objects["internal"],
    ax=[axs[0], axs[1]],
    orientation="vertical",
    location="right",
    aspect=25,
    ticks=[z_int_pie[0], (z_int_pie[0] + z_int_pie[1]) / 2, z_int_pie[1]],
    pad=0.02,
)

# Colorbar for external mics (channels 2, 3)
cbar2 = fig.colorbar(
    img_objects["external"],
    ax=[axs[2], axs[3]],
    orientation="vertical",
    location="right",
    aspect=25,
    ticks=[z_int_mic[0], (z_int_mic[0] + z_int_mic[1]) / 2, z_int_mic[1]],
    pad=0.02,
)
cbar2.ax.set_ylabel("Power [dB]")

# Colorbar for last external mic (channel 4)
cbar3 = fig.colorbar(
    img_objects["last"],
    ax=axs[4],
    orientation="vertical",
    location="right",
    aspect=12.5,
    ticks=[z_ext_mic[0], (z_ext_mic[0] + z_ext_mic[1]) / 2, z_ext_mic[1]],
    pad=0.02,
)
fig.suptitle("Noise 90 dBA")
# %%
