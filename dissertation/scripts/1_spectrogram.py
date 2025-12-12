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

import cblind as cb

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

record = db.session.get(spidb.Record, 950)

audios = db.get_audios(
    start=record.start, end=record.end, sensor=record.sensor, channels=[0, 1, 2, 3]
)

labels = [f"Ch. {i}" for i in range(len(audios))]

fig, ax = visualizer.spectrogram_display(
    audios,
    labels,
    time_format="seconds",
    showscale="right",
    zmin=-140,
    zmax=-80,
    color_map=cb.cbmap("cb.solstice"),
)
for a in ax:
    a.set_ylim(0, 12000)
    a.set_yticks([0, 6000, 12000])
# change figure size to 6x3 inches
fig.set_size_inches(6, 3)
fig.savefig(r"projects/Dissertation/dissertation/figures/1_spectrogram.pdf", dpi=300)
