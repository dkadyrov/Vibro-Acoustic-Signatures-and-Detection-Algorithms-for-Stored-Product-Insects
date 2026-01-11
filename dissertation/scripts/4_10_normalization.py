# %%

import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import numpy as np

from spidb import spidb
from aspids_tools import normalization

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

target = {
    "target": "Mealworm",
    "material": "Wheat Groats",
    "test": 93,
    "iteration": 12,
    "channel": 2,
    "record": 1007,
    "amplitude": 5000,
    "amplitude2": 10000,
}


# %%
record = db.session.get(spidb.Record, target["record"])
audio = db.get_audio(
    start=record.start,
    end=record.end,
    sensor=record.sensor,
    channel_number=target["channel"],
)

f, p = signal.welch(
    audio.data.signal,
    fs=audio.sample_rate,
    nperseg=1024,
    noverlap=512,
    window="blackmanharris",
    scaling="spectrum",
)
p = 10 * np.log10(p)

fig, ax = plt.subplots()
ax.plot(f, p, label="Signal")
ax.plot(f, noise[f"{target['channel']}_db"], label="Self-noise")

audio.bandpass_filter(500, 8000, 10, overwrite=True)
audio.envelope(overwrite=True)

level = audio.data.signal < 5 * np.median(audio.data.signal)
n = audio.data[audio.data.signal < level]

f, n = signal.welch(
    n.signal,
    fs=audio.sample_rate,
    nperseg=1024,
    window="blackmanharris",
    scaling="spectrum",
)
n = 10 * np.log10(n)

ax.plot(f, n, label="Median Noise")
ax.set_xlim(0, 8000)
ax.set_ylim(-140, -40)
ax.set_yticks([-140, -90, -40])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
ax.legend(loc="upper right", ncols=3)

fig.savefig(
    r"projects/Dissertation/dissertation/figures/10_median_normalization.pdf",
    dpi=300,
)
# %%
