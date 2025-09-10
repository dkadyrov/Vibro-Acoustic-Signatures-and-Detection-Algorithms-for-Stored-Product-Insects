# %%

import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal 
import numpy as np 

from spidb import normalization, spidb

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

targets = [
    {
        "target": "Darkling beetle",
        "material": "Rice",
        "test": 83,
        "iteration": 2,
        "channel": 0,
        "record": 950,
        "amplitude": 20000,
        "amplitude2": 20000
    },
    {
        "target": "Mealworm",
        "material": "Wheat Groats",
        "test": 93,
        "iteration": 12,
        "channel": 2,
        "record": 1007,
        "amplitude": 10000,
        "amplitude2": 10000
    },
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "test": 163,
        "iteration": 6,
        "channel": 0,
        "record": 2236,
        "amplitude": 300,
        "amplitude2": 300
    },
    {
        "target": "Bean beetle",
        "material": "Oatmeal",
        "test": 7,
        "iteration": 1,
        "channel": 1,
        "record": 40,
        "amplitude": 1500,
        "amplitude2": 1500
    },
    {
        "target": "Noise",
        "material": "60 dBA",
        "channel": 0,
        "record": 460,
        "amplitude": 300,
        "amplitude2": 10
    },
    {
        "target": "Noise",
        "material": "90 dBA",
        "channel": 0,
        "record": 463,
        "amplitude": 1500,
        "amplitude2": 40
    },
]

# %%
filts = [[500, 6000],[1565, 6000]]

for f, filt in enumerate(filts):

    low = filt[0]
    high = filt[1]

    for i, t in enumerate(targets[-1:]):
        record = db.session.get(spidb.Record, t["record"])

        audio = db.get_audio(
            start=record.start,
            end=record.end,
            sensor=record.sensor,
            channel_number=t["channel"],
        )

        audio.fade_in(1, overwrite=True)
        audio.fade_out(1, overwrite=True)

        audio.bandpass_filter(filt[0], filt[1], 10, overwrite=True)

        audio.envelope(overwrite=True)

        channel = record.sensor.channels[t["channel"]]
        audio = normalization.noise_normalize(db, audio, channel=channel, filter="bandpass", low=low, high=high, coefficient="set")

        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.plot(audio.data.seconds, audio.data.signal)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Normalized\nAmplitude")
        ax.set_xlim(0, 60)
        if f == 0:
            ax.set_ylim(0, t["amplitude"])
            ax.set_yticks([0, round(t["amplitude"]/2), t["amplitude"]])
        else: 
            ax.set_ylim(0, t["amplitude2"])
            ax.set_yticks([0, round(t["amplitude2"]/2), t["amplitude2"]])

        fig.savefig(f"projects/Dissertation/proposal/figures/normalization/{t['target']}_{t['material']}_waveform_normalized_small ({filt[0]}-{filt[1]}).pdf", dpi=300)
#%%
record = db.session.get(spidb.Record, targets[1]["record"])
audio = db.get_audio(
    start=record.start,
    end=record.end,
    sensor=record.sensor,
    channel_number=targets[1]["channel"],
)

f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")
p = 10 * np.log10(p)

fig, ax = plt.subplots()
ax.plot(f, p, label="Signal")
ax.plot(f, noise[f"{targets[1]['channel']}_db"], label="Self-noise")

# audio.fade_in(1, overwrite=True)
# audio.fade_out(1, overwrite=True)
audio.bandpass_filter(500, 8000, 10, overwrite=True)
audio.envelope(overwrite=True)

level = audio.data.signal < 5*np.median(audio.data.signal)
n = audio.data[audio.data.signal < level]

f, n = signal.welch(n.signal, fs=audio.sample_rate, nperseg=1024, window="blackmanharris", scaling="spectrum")
n = 10 * np.log10(n)

ax.plot(f, n, label="Median Noise")
ax.set_xlim(0, 8000)
ax.set_ylim(-140, -40)
ax.set_yticks([-140, -90, -40])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
ax.legend(loc="upper right", ncols=3)

fig.savefig(f"projects/Dissertation/proposal/figures/normalization/normalization_comparison.pdf", dpi=300)
#%%