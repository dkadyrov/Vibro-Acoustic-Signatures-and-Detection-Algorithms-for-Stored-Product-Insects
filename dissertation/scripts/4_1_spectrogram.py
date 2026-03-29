import cblind
from aspids_tools import visualizer
from matplotlib import pyplot as plt
import matplotlib

from spidb import spidb

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
    color_map=matplotlib.colormaps.get_cmap("cb.solstice"),
)
for a in ax:
    a.set_ylim(0, 12000)
    a.set_yticks([0, 6000, 12000])
# change figure size to 6x3 inches
fig.set_size_inches(6, 4)
fig.savefig(r"projects/Dissertation/dissertation/figures/1_spectrogram.pdf", dpi=300)
