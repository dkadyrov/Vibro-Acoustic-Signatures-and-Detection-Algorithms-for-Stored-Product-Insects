# %%
from spidb import spidb, detection, visualizer, normalization
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dankpy import document, acoustics
from scipy import signal

plt.style.use("dankpy.styles.presentation")

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
#%%
frequency = np.linspace(500, 3000, 21)

for f in frequency:
    for channel in channels: 
        channel = db.get_channel(channel, sensor=sensor)
        channel.gain = normalization.noise_coefficient(db, sensor, channel, filter="bandpass", low=f, high=8000)

    for e, record in enumerate(records):
        r = db.session.get(spidb.Record, record["record"])

        result = {
            "Record ID": r.id,
            "Subject": record["target"],
            "Frequency": f
        }

        audios = detection.retreive_acoustic_data(db, r.sensor, r.start, r.end, channels=channels)

        for c, audio in enumerate(audios):
            channel = r.sensor.channels[channels[c]]

            nspa = normalization.calculate_nspa(audio, filter="bandpass", low=f, high=8000, normalize="noise", channel=channel, ratio=True)
            
            result[f"Channel {channels[c]}"] = nspa
        
        results.append(result)
results = pd.DataFrame(results)
#%%
diff = results.iloc[:,2:].groupby("Frequency").diff().dropna()
diff["Frequency"] = results["Frequency"].unique()
# %%
fig, ax = plt.subplots()
for c in channels: 
    if c < 4:
        ax.scatter(diff["Frequency"], diff[f"Channel {c}"], label=f"Ch. {c}")
    else:
        ax.scatter(diff["Frequency"], diff[f"Channel {c}"], label=f"Ch. {c}", color="black", marker="x", s=50)
ax.set_xlabel("Lower Frequency [Hz]")
ax.set_ylabel("NSPA Difference [dB]", fontsize=9)
ax.set_xlim(500, 3000)
ax.set_ylim(-40, 40)
ax.set_yticks([-40, 0, 40])
ax.legend(loc="upper right", ncol=5, markerscale=0.5)
fig.savefig("projects/Dissertation/proposal/figures/external_noise/lower_frequency_nspa_difference.pdf", bbox_inches="tight", dpi=300)

#%%
results = []

frequency = np.linspace(2000, 8000, 21)

for f in frequency:
    for channel in channels: 
        channel = db.get_channel(channel, sensor=sensor)
        channel.gain = normalization.noise_coefficient(db, sensor, channel, filter="bandpass", low=1625, high=f)

    for e, record in enumerate(records):
        r = db.session.get(spidb.Record, record["record"])

        result = {
            "Record ID": r.id,
            "Subject": record["target"],
            "Frequency": f
        }

        audios = detection.retreive_acoustic_data(db, r.sensor, r.start, r.end, channels=channels)

        for c, audio in enumerate(audios):
            channel = r.sensor.channels[channels[c]]

            nspa = normalization.calculate_nspa(audio, filter="bandpass", low=1625, high=f, normalize="noise", channel=channel, ratio=True)
            
            result[f"Channel {channels[c]}"] = nspa
        
        results.append(result)
results = pd.DataFrame(results)

diff = results.iloc[:,2:].groupby("Frequency").diff().dropna()
diff["Frequency"] = results["Frequency"].unique()
#%%
fig, ax = plt.subplots()
for c in channels: 
    if c < 4:
        ax.scatter(diff["Frequency"], diff[f"Channel {c}"], label=f"Ch. {c}")
    else:
        ax.scatter(diff["Frequency"], diff[f"Channel {c}"], label=f"Ch. {c}", color="black", marker="x", s=50)
ax.set_xlabel("Upper Frequency [Hz]")
ax.set_ylabel("NSPA Difference [dB]", fontsize=9)
ax.set_xlim(2000, 8000)
ax.set_ylim(-60, 20)
ax.set_yticks([-60, -20, 20])
# ax.legend(loc="upper right", ncol=5, markerscale=0.5)
fig.savefig("projects/Dissertation/proposal/figures/external_noise/upper_frequency_nspa_difference.pdf", bbox_inches="tight", dpi=300)
# %%
