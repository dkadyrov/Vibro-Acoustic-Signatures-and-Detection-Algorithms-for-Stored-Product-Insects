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

import cblind as cb

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

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

fig, ax = plt.subplots(figsize=(6, 1.75))

for i, t in enumerate(targets):
    record = db.session.get(spidb.Record, t["record"])

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
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

    ax.plot(
        f,
        p,
        label=f"{lookup.lookup(key=t['target'], latex=True, min=True)} ({t['material']})",
        color=color.colors[i],
    )
ax.plot(
    noise["frequency"],
    noise["average_db"],
    color="black",
    linestyle="solid",
    label="Reference",
    zorder=10,
)
ax.set_xlim(0, 8000)
ax.set_ylim(-125, -25)
ax.set_yticks([-125, -75, -25])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
ax.legend(
    bbox_to_anchor=(0.0, 1.10, 1, 0),
    loc="lower center",
    ncol=4,
    columnspacing=1,
    markerscale=0.5,
    fontsize=8,
)
fig.savefig("projects/Dissertation/proposal/figures/2_spectra_comparison.pdf", dpi=300)
# %%
