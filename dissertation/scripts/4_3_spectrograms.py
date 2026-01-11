# %%
from archive import lookup
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal

from dankpy import color, dt

from spidb import spidb
from aspids_tools import visualizer
import matplotlib.pyplot as plt

import cblind as cb

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")


targets = [
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "test": 163,
        "iteration": 6,
        "channel": 0,
        "record": 2236,
    },
    {
        "target": "Bean beetle",
        "material": "Oatmeal",
        "test": 7,
        "iteration": 1,
        "channel": 1,
        "record": 40,
    },
]

alpha = ["a", "b"]

for i, t in enumerate(targets):
    record = db.session.get(spidb.Record, t["record"])
    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    fig, ax = audio.plot_spectrogram(
        window_size=1024,
        nperseg=1024,
        nfft=1024,
        noverlap=512,
        zmin=-140,
        zmax=-80,
        time_format="seconds",
        showscale="right",
        cmap=cb.cbmap("cb.solstice"),
    )

    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 4000, 8000])
    # add label to legend
    ax.plot([], [], " ", label=f"Ch. {t['channel']}")
    ax.legend(handlelength=0, handletextpad=0)

    fig.savefig(
        r"projects\Dissertation\dissertation\figures/3{}_spectrogram.pdf".format(
            alpha[i]
        ),
        dpi=300,
    )
# %%
