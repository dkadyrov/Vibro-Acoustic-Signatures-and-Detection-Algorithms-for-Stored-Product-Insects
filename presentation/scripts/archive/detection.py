# %%
from spidb import spidb, detection, visualizer, normalization
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dankpy import document, acoustics
from scipy import signal

plt.style.use("dankpy.styles.stevens_presentation")

db = spidb.Database(r"data/spi.db")

#%%
records = [    
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "record": 2236,
        "channel": 2,
    },
    {
        "target": "Noise (No insect)",
        "material": "Rice",
        "record": 461,
        "channel": 0,
    }
]

results = []

channels = [0, 1, 2, 3, 7]
sensor = db.session.get(spidb.Sensor, 1)
frequency = np.linspace(5000, 6000, 21)
#%%
for record in records:
    record = db.session.get(spidb.Record, record["record"])
    audios = detection.retreive_acoustic_data(db, record.sensor, record.start, record.end, channels=[0, 1, 2, 3, 7])
    data, acoustic_data = detection.acoustic_detection(audios, IT=7, ET=5, DR=9, NR=30, internal_filt=[1565, 6000], external_filt=1565, scaling_i=4, scaling_e=4, return_audio=True)

    fig, ax = detection.acoustic_detection_display(data, acoustic_data, time_format="seconds", latex=False, spl=False)
    fig.set_size_inches(5.67, 4.24)
# %%
