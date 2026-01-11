# %%
from archive import lookup
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal

from dankpy import color

from spidb import spidb
import matplotlib.pyplot as plt

from aspids_tools import normalization

import cblind as cb

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
# %%
target = {
    "target": "Mealworm",
    "material": "Wheat Groats",
    "test": 93,
    "iteration": 12,
    "channel": 2,
    "record": 1007,
}

record = db.session.get(spidb.Record, target["record"])

audio = db.get_audio(
    start=record.start,
    end=record.end,
    sensor=record.sensor,
    channel_number=target["channel"],
)

fig, ax = plt.subplots()
ax.plot(
    audio.data.seconds,
    audio.data.signal,
    label=f"Ch. {target['channel']}",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.set_ylim(-0.3, 0.3)
ax.set_yticks([-0.3, 0, 0.3])
ax.set_xlim(0, 60)
ax.legend(loc="upper right", handlelength=0, handletextpad=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\5_rawwaveform.pdf",
    dpi=300,
)

audio.bandpass_filter(500, 6000, 10, overwrite=True)

fig, ax = plt.subplots()
ax.plot(
    audio.data.seconds,
    audio.data.signal,
    label=f"Ch. {target['channel']}",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.set_ylim(-0.3, 0.3)
ax.set_yticks([-0.3, 0, 0.3])
ax.set_xlim(0, 60)
ax.legend(loc="upper right", handlelength=0, handletextpad=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\6_waveform_filtered.pdf",
    dpi=300,
)
# %%
audio.envelope(overwrite=True)
fig, ax = plt.subplots()
ax.plot(
    audio.data.seconds,
    audio.data.signal,
    label=f"Ch. {target['channel']}",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.set_ylim(0, 0.3)
ax.set_yticks([0, 0.15, 0.3])
ax.set_xlim(0, 60)
ax.legend(loc="upper right", handlelength=0, handletextpad=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\7_waveform_envelope.pdf",
    dpi=300,
)
# %%
audio = normalization.noise_normalize(
    db,
    audio,
    channel=record.sensor.channels[target["channel"]],
    filter="bandpass",
    low=500,
    high=6000,
    coefficient="set",
    fade_time=2,
)
# %%
fig, ax = plt.subplots()
ax.plot(
    audio.data.seconds,
    audio.data.signal,
    label=f"Ch. {target['channel']}",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized\nAmplitude")
ax.set_ylim(0, 6000)
ax.set_yticks([0, 3000, 6000])
ax.set_xlim(0, 60)
ax.legend(loc="upper right", handlelength=0, handletextpad=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\9_waveform_normalized.pdf",
    dpi=300,
)
# %%
