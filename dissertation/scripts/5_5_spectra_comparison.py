# %%
from spidb import lookup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from dankpy import color 

from spidb import spidb

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
# %%
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
        "material": "Wheat Groats",
        "test": 7,
        "iteration": 1,
        "channel": 0,
        "record": 329,
    },
    {
        "target": "Noise",
        "material": "90 dBA",
        "channel": 0,
        "record": 463,
        "amplitude": 500,
    },
]

fig, ax = plt.subplots()
for i, target in enumerate(targets):
    record = db.session.get(spidb.Record, target["record"])
    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=target["channel"],
    )

    # Convert to a NumPy array so SciPy does positional slicing reliably.
    samples = audio.data.signal.to_numpy(copy=False)

    f, p = signal.welch(
        samples,
        fs=audio.sample_rate,
        nperseg=1024,
        noverlap=512,
        window="blackmanharris",
        scaling="spectrum",
    )

    p = 10 * np.log10(p)

    if target["target"] == "Noise":
        ax.plot(
            f, p, label=f"Noise ({target['material']})", color="black", linestyle="--"
        )
    else:
        ax.plot(
            f,
            p,
            label=f"{lookup.lookup(key=target['target'], latex=True, min=True)} ({target['material']})",
            color=color.colors[i],
        )

ax.plot(
    noise["frequency"],
    10 * np.log10(noise["0"]),
    label="Reference",
    color="black",
    linestyle="-",
)

ax.set_xlim(0, 8000)
ax.set_ylim(-125, -25)
ax.set_yticks([-125, -75, -25])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
# put the legend on top
ax.legend(loc="upper right", ncol=2)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\5_6_spectra_comparison.pdf",
    dpi=300,
)
# %%
