# %%

from datetime import timedelta
from email.mime import audio
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dankpy import acoustics, color, document
from scipy import signal

from spidb import detection, lookup, normalization, spidb, visualizer

plt.style.use("dankpy.styles.latex")

db = spidb.Database(r"data/spi.db")

samples = [1562, 1618, 1131, 7242, 7747, 7124, 9339, 9387, 12859, 17843, 17340, 16849]

noises = [460,461,462, 463]

freqs = [[500, 6000], [1565, 6000]]

sensor = db.session.get(spidb.Sensor, 1)

data = []

for freq in freqs: 
    low, high = freq
    for channel in sensor.channels:
        channel.gain = normalization.noise_coefficient(db, sensor, channel, "bandpass", low, high, order=10)

    for sample in samples: 
        sample = db.session.get(spidb.Sample, sample)

        audio = db.get_audio(sample.record.start, sample.record.end, sensor=sample.sensor, channel=sample.channel)

        nspa_sample = normalization.calculate_nspa(audio, filter="bandpass", low=low, high=high, normalize="noise", channel=sample.channel)

        data.append({
            "sample": sample.id,
            "subject": sample.subject.id,
            "material": sample.material.id,
            "low": low,
            "high": high,
            "nspa": nspa_sample,
        })

    for noise in noises:
        record = db.session.get(spidb.Record, noise)

        audio = db.get_audio(record.start, record.end, sensor=record.sensor, channel=record.sensor.channels[0])

        nspa_noise = normalization.calculate_nspa(audio, filter="bandpass", low=low, high=high, normalize="noise", channel=record.sensor.channels[0])

        data.append({
            "sample": record.id,
            "subject": "noise",
            "material": record.external_spl,
            "low": low,
            "high": high,
            "nspa": nspa_noise
        })

#%%
data = pd.DataFrame(data)

# average the rows that don't have the subject noise but keep the noise rows
table = data[data["subject"] != "noise"].groupby(["low", "high"]).mean().reset_index()

# add the noise rows back in
table = pd.concat([table, data[data["subject"] == "noise"]])
# %%
table["freq"] = table.apply(lambda row: f"{row['low']}-{row['high']}", axis=1)
# %%
# make a pivot table where the index is the material and the columns are the freq column and teh values are the nspa
pivot = table.pivot_table(index="material", columns="freq", values="nspa").reset_index()
pivot.to_latex(index=False)
# %%
