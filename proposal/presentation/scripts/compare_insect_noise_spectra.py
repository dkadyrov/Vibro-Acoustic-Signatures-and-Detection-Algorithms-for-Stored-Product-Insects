# %%
from archive import lookup
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal 

from dankpy import color, dt

from spidb import spidb, visualizer, detection
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
# %%
targets = [
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "name": "$\mathit{T. confusum}$ (Flour)",
        "test": 163,
        "iteration": 6,
        "channel": 0,
        "record": 2236
    },
    {
        "target": "Bean beetle",
        "material": "Oatmeal",
        "name": "$\mathit{C. maculatus}$ (Oatmeal)",
        "test": 7,
        "iteration": 1,
        "channel": 1,
        "record": 40
    },
    # {
    #     "target": "Noise",
    #     "material": "90 dBA",
    #     "name": "Noise (90 dBA)",
    #     "channel": 0,
    #     "record": 463,
    #     "amplitude": 500,
    # },
    {
        "target": "Noise",
        "material": "90 dBA",
        "name": "Noise (90 dBA)",
        "channel": 1,
        "record": 580,
        "amplitude": 500,
    },
]

fig, ax = plt.subplots(figsize=(11.71,4.24))
for i, target in enumerate(targets):
    record = db.session.get(spidb.Record, target["record"])
    audio = db.get_audio(start=record.start, end=record.end, sensor=record.sensor, channel_number=target["channel"])

    f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")

    p = 10 * np.log10(p)

    if target["target"] == "Noise":
        ax.plot(f, p, label=target['name'], color="black", linestyle="-")
    else:
        ax.plot(f, p, label=target['name'], color=color.colors[i])

ax.plot(noise["frequency"], 10 * np.log10(noise["0"]), label="Reference", color="black", linestyle="--")

ax.set_xlim(0, 8000)
ax.set_ylim(-125, -25)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
# put the legend on top
ax.legend(loc="upper right")
# fig.savefig(r"projects/Dissertation/proposal/figures/external_noise/spectra_comparison.pdf", dpi=300)
#%%
record = db.session.get(spidb.Record, targets[-1]["record"])

fig, ax = visualizer.waveform_display(db, start=record.start, end=record.end, sensor=record.sensor, time_format="seconds", external_spl=True, envelope=True, normalize="noise", filter=[1565, 6000], size=(5.67,4.21))
for i, a in enumerate(ax):
    # if i < len(ax)-1:
    a.set_ylim(0, 50)
    # a.set_yticks([0, 50])
    # else:
        # a.set_ylim(0, 500)
        # a.set_yticks([0, 500])

    # a.set_yticks([0, 2500, 5000])
    # update legend in a to have smaller font 
    a.legend(fontsize=10, loc="upper right", handlelength=0, handletextpad=0)
print(f"Channel 0: {record.classifications[0].classification}\nChannel 1: {record.classifications[1].classification}\nChannel 2: {record.classifications[2].classification}\nChannel 3: {record.classifications[3].classification}")

print(np.std([float(record.classifications[0].classification), float(record.classifications[1].classification), float(record.classifications[2].classification), float(record.classifications[3].classification)]))

