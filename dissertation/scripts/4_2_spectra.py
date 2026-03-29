import matplotlib.pyplot as plt
import pandas as pd
from aspids_tools import visualizer
from dankpy import color

from spidb import spidb

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

record = db.session.get(spidb.Record, 950)

audios = db.get_audios(
    start=record.start, end=record.end, sensor=record.sensor, channels=[0, 1, 2, 3]
)

fig, ax = visualizer.spectra_display(audios)
for i, line in enumerate(ax.lines):
    line.set_color(color.colors[i])
    line.set_label(f"Ch. {i}")
ax.plot(
    noise["frequency"],
    noise["average_db"],
    color="black",
    linestyle="solid",
    label="Reference",
    zorder=10,
)
ax.set_xlim(0, 12000)
ax.legend(loc="upper right", ncols=5)
ax.set_ylabel("Spectral Power [dB]", fontsize=8)
ax.set_ylim(-125, -25)
ax.set_yticks([-125, -75, -25])

fig.savefig(
    r"projects\Dissertation\dissertation\figures\2_spectra.pdf",
    dpi=300,
)
