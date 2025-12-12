# %%
from spidb import spidb, normalization, visualizer, detection
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dankpy import acoustics, color
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal

plt.style.use("dankpy.styles.stevens_presentation")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")
#%%
# events = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49] 
# events = [41, 42, 43, 44]
# events = [44]
targets = [
    # {
    #     "target": "Confused flour beetle",
    #     "material": "Flour",
    #     "name": "$\mathit{T. confusum}$ (Flour)",
    #     "test": 163,
    #     "iteration": 6,
    #     "channel": 0,
    #     "record": 2236
    # },
    # {
    #     "target": "Bean beetle",
    #     "material": "Oatmeal",
    #     "name": "$\mathit{C. maculatus}$ (Oatmeal)",
    #     "test": 7,
    #     "iteration": 1,
    #     "channel": 1,
    #     "record": 40
    # },
    {
        "target": "Noise",
        "material": "90 dBA",
        "name": "Noise",
        "channel": 0,
        "record": 463,
        "amplitude": 1000,
    },
]

for i, target in enumerate(targets): 
    event = db.session.get(spidb.Record, target["record"])

    start = event.start
    end = start + timedelta(seconds=60)

    # fig, ax = visualizer.spectrogram_display(db, start=start, end=end, sensor=event.sensor, time_format="seconds", section="minimal", showscale="right", zmin=-140, zmax=-80, external_spl=True,  size=(11.71,4.21))
    # for a in ax: 
    #     a.set_ylim(0, 8000)
    #     a.set_yticks([0, 4000, 8000])
    # fig.suptitle("Noise - No Insect")


    fig, ax = visualizer.waveform_display(db, start=start, end=end, sensor=event.sensor, time_format="seconds", external_spl=False, envelope=True, normalize="noise", filter=[1565, 6000], size=(5.67,4.21))
    for i, a in enumerate(ax):
        if i < len(ax)-1:
            a.set_ylim(0, 50)
            a.set_yticks([0, 50])
        else:
            a.set_ylim(0, 500)
            a.set_yticks([0, 500])

        # a.set_yticks([0, 2500, 5000])
        # update legend in a to have smaller font 
        a.legend(fontsize=10, loc="upper right", handlelength=0, handletextpad=0)
    fig.suptitle(target["name"])
    break
    # fig.savefig(f"projects/Dissertation/proposal/figures/external_noise/waveform_{event.id}.pdf", dpi=300)

    #%%