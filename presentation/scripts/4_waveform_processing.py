# %%
from spidb import lookup
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal

from dankpy import color, dt

from spidb import spidb
from aspids_tools import visualizer
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
# %%
targets = [
    {
        "target": "Darkling beetle",
        "material": "Rice",
        "test": 83,
        "iteration": 2,
        "channel": 0,
        "record": 950,
    },
]

for i, t in enumerate(targets[:]):
    record = db.session.get(spidb.Record, t["record"])

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    audio.fade_in(1, overwrite=True)
    audio.fade_out(1, overwrite=True)

    fig, ax = plt.subplots(figsize=(3.61, 4.21))
    ax.plot(
        audio.data.seconds,
        audio.data.signal,
        label=f"Ch. {t['channel']}",
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    # ax.set_yticks([-0.5, 0, 0.5])
    # ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)
    ax.set_title("Raw Signal")

    audio.bandpass_filter(500, 6000, 10, overwrite=True)
    fig, ax = plt.subplots(figsize=(3.61, 4.21))
    ax.plot(
        audio.data.seconds,
        audio.data.signal,
        label=f"Ch. {t['channel']}",
    )
    ax.set_xlim(0, 60)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)
    ax.set_title("Filtered Signal")

    audio.envelope(overwrite=True)
    fig, ax = plt.subplots(figsize=(3.61, 4.21))
    ax.plot(
        audio.data.seconds,
        audio.data.signal,
        label=f"Ch. {t['channel']}",
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Filtered and Enveloped Signal")
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)

# %%
