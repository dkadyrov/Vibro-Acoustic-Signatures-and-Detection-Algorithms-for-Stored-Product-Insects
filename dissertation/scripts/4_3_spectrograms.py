import cblind
import matplotlib
import matplotlib.pyplot as plt

from spidb import spidb

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
        start=record.start, end=record.end, sensor=record.sensor, channel_number=t["channel"],
    )

    fig, ax = audio.plot_spectrogram(
        window_size=1024, nperseg=1024, nfft=1024, noverlap=512, zmin=-140, zmax=-80, time_format="seconds", showscale="right", cmap=matplotlib.colormaps.get_cmap("cb.solstice")
    )

    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 4000, 8000])
    ax.plot([], [], " ", label=f"Ch. {t['channel']}")
    ax.legend(handlelength=0, handletextpad=0)

    fig.savefig(r"projects\Dissertation\dissertation\figures/3{}_spectrogram.pdf".format(alpha[i]), dpi=300)
