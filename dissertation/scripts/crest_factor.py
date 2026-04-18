# %%
from copy import deepcopy
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dankpy import color
from scipy.signal import welch
from sonicdb import audio
from spidb import spidb

from aspids_tools import normalization, processing

import cblind as cb  # type: ignore  # noqa: F401

plt.style.use("dankpy.styles.mdpi")
# %%
db = spidb.Database(r"data/spi.db")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

noise_record = db.session.get(spidb.Record, 464)


noise = db.get_audio(
    sensor=noise_record.sensor,
    start=noise_record.start
    + timedelta(seconds=40),  # - timedelta(seconds=20),  # + timedelta(seconds=41),
    end=noise_record.end,  # - timedelta(seconds=20),  # - timedelta(seconds=7),
    channel_number=0,
)

noise = processing.process_signal(
    db, noise, channel=noise_record.sensor.channels[0]
)
#%%
peak = np.max(np.abs(noise.data.signal))
rms = np.sqrt(np.mean(noise.data.signal**2))
crest_factor = 10 * np.log10(peak / rms)
cr = peak/rms

min_threshold = 0.5 * np.max(noise.data.signal)
max_threshold = 0.9 * np.max(noise.data.signal)

cutoff = noise.data.signal[noise.data.signal >= min_threshold]
cutoff = cutoff[cutoff <= max_threshold]

rms = np.sqrt(np.mean(cutoff**2))
amp = 20 * np.log10(rms)


fig, ax = plt.subplots()
ax.plot(noise.data.seconds, noise.data.signal)
ax.axhline(cr, color="red", linestyle="--",  label=f"CR: {crest_factor:.0f} dB")
ax.axhline(rms, color="blue", linestyle="--",  label=f"NSPA: {amp:.0f} dB")
ax.legend(loc="upper right")
ax.set_xlim(0, 20)
ax.set_ylim(0, None)

# %%
noise_record = db.session.get(spidb.Record, 468)


noise = db.get_audio(
    sensor=noise_record.sensor,
    start=noise_record.start
    + timedelta(seconds=40),  # - timedelta(seconds=20),  # + timedelta(seconds=41),
    end=noise_record.end,  # - timedelta(seconds=20),  # - timedelta(seconds=7),
    channel_number=0,
)

noise = processing.process_signal(db, noise, channel=noise_record.sensor.channels[0])

peak = np.max(np.abs(noise.data.signal))
rms = np.sqrt(np.mean(noise.data.signal**2))
crest_factor = 10 * np.log10(peak / rms)

print(f"Crest Factor (White Noise, 100 dbA): {crest_factor:.0f} dB")
#%%
noise_record = db.session.get(spidb.Record, 455)


noise = db.get_audio(
    sensor=noise_record.sensor,
    start=noise_record.start
    + timedelta(seconds=40),  # - timedelta(seconds=20),  # + timedelta(seconds=41),
    end=noise_record.end,  # - timedelta(seconds=20),  # - timedelta(seconds=7),
    channel_number=0,
)

noise = processing.process_signal(db, noise, channel=noise_record.sensor.channels[0])

peak = np.max(np.abs(noise.data.signal))
rms = np.sqrt(np.mean(noise.data.signal**2))
crest_factor = 10 * np.log10(peak / rms)

cr = peak/rms

min_threshold = 0.5 * np.max(noise.data.signal)
max_threshold = 0.9 * np.max(noise.data.signal)

cutoff = noise.data.signal[noise.data.signal >= min_threshold]
cutoff = cutoff[cutoff <= max_threshold]

rms = np.sqrt(np.mean(cutoff**2))
amp = 20 * np.log10(rms)


fig, ax = plt.subplots()
ax.plot(noise.data.seconds, noise.data.signal)
ax.axhline(cr, color="red", linestyle="--",  label=f"CR: {crest_factor:.0f} dB")
ax.axhline(rms, color="blue", linestyle="--",  label=f"NSPA: {amp:.0f} dB")
ax.legend(loc="upper right")
ax.set_xlim(0, 20)
ax.set_ylim(0, None)

print(f"White Noise (100 dBA) Crest Factor: {crest_factor:.0f} dB")
#%%
noise_record = db.session.get(spidb.Record, 454)


noise = db.get_audio(
    sensor=noise_record.sensor,
    start=noise_record.start
    + timedelta(seconds=40),  # - timedelta(seconds=20),  # + timedelta(seconds=41),
    end=noise_record.end,  # - timedelta(seconds=20),  # - timedelta(seconds=7),
    channel_number=0,
)

noise = processing.process_signal(db, noise, channel=noise_record.sensor.channels[0])

peak = np.max(np.abs(noise.data.signal))
rms = np.sqrt(np.mean(noise.data.signal**2))
crest_factor = 10 * np.log10(peak / rms)
cr = peak/rms

min_threshold = 0.5 * np.max(noise.data.signal)
max_threshold = 0.9 * np.max(noise.data.signal)

cutoff = noise.data.signal[noise.data.signal >= min_threshold]
cutoff = cutoff[cutoff <= max_threshold]

rms = np.sqrt(np.mean(cutoff**2))
amp = 20 * np.log10(rms)


fig, ax = plt.subplots()
ax.plot(noise.data.seconds, noise.data.signal)
ax.axhline(cr, color="red", linestyle="--",  label=f"CR: {crest_factor:.0f} dB")
ax.axhline(rms, color="blue", linestyle="--",  label=f"NSPA: {amp:.0f} dB")
ax.legend(loc="upper right")
ax.set_xlim(0, 20)
ax.set_ylim(0, None)

print(f"White Noise (90 dBA) Crest Factor: {crest_factor:.0f} dB")
print(f"White Noise (90 dBA) NSPA: {amp:.0f} dB")
# %%
