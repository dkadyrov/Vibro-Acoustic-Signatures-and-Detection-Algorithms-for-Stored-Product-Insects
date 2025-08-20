# %%
from spidb import spidb, normalization, visualizer, detection
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dankpy import acoustics, color
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal

plt.style.use("dankpy.styles.latex")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")
#%%
# events = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49] 
events = [41, 42, 43, 44]
data = [] 

for i, event in enumerate(events): 
    event = db.session.get(spidb.Event, event)

    start = event.start
    end = start + timedelta(seconds=60)

    fig, ax = visualizer.spectrogram_display(db, start=start, end=end, sensor=event.sensor, time_format="seconds", section="minimal", showscale="right", zmin=-140, zmax=-80, external_spl=True,     size=(6,3.25))
    for a in ax: 
        a.set_ylim(0, 8000)
        a.set_yticks([0, 4000, 8000])
    fig.savefig(f"projects/Dissertation/proposal/figures/external_noise/spectrogram_{event.id}.pdf", dpi=300)

    fig, ax = visualizer.waveform_display(db, start=start, end=end, sensor=event.sensor, time_format="seconds", external_spl=True, envelope=True, normalize="noise", filter=True, size=(6,3.25))
    for a in ax: 
        a.set_ylim(0, 250)
        a.set_yticks([0, 125, 250])
    fig.savefig(f"projects/Dissertation/proposal/figures/external_noise/waveform_{event.id}.pdf", dpi=300)

    #%%