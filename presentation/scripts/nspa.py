# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from spidb import normalization, spidb

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

targets = [
    {
        "target": "Darkling beetle",
        "name": "T.\ molitor",
        "material": "Rice",
        "test": 83,
        "iteration": 2,
        "channel": 0,
        "record": 950,
        "range": [0, 7500, 15000]
    },
    {
        "target": "Mealworm",
        "name": "T.\ molitor",
        "material": "Wheat Groats",
        "test": 93,
        "iteration": 12,
        "channel": 2,
        "record": 1007,
        "range": [0, 3000, 6000]
    },
    {
        "target": "Confused flour beetle",
        "name": "T.\ confusum",
        "material": "Flour",
        "test": 163,
        "iteration": 6,
        "channel": 0,
        "record": 2236,
        "range": [0, 150, 300]
    },
    {
        "target": "Bean beetle",
        "name": "C. maculatus",
        "material": "Oatmeal",
        "test": 7,
        "iteration": 1,
        "channel": 1,
        "record": 40,
        "range": [0, 500, 1000]
    },
]

# %%
for i, t in enumerate(targets):
    record = db.session.get(spidb.Record, t["record"])
    channel = record.sensor.channels[t["channel"]]

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    audio.fade_in(1, overwrite=True)
    audio.fade_out(1, overwrite=True)

    audio.bandpass_filter(500, 6000, 10, overwrite=True)

    channel.gain = normalization.noise_coefficient(db, channel.sensor, channel=channel, filter="bandpass", low=500, high=6000, order=10)

    audio.envelope(overwrite=True)
    audio = normalization.noise_normalize(audio, channel=channel)

    min_threshold = 0.5 * np.max(audio.data.signal)
    max_threshold = 0.9 * np.max(audio.data.signal)

    cutoff = audio.data.signal[audio.data.signal >= min_threshold]
    cutoff = cutoff[cutoff <= max_threshold]

    rms = np.sqrt(np.mean(cutoff**2))
    amp = 20 * np.log10(rms)

    fig, ax = plt.subplots(figsize=(3.61, 4.21))
    ax.plot(audio.data.seconds, audio.data.signal)


    # make a dashed red line at the RMS value, round it to 2 decimal places
    ax.axhline(rms, color="red", linestyle="--", label=f"NSPA ({amp:.2f} dB)", zorder=10)
 
    # make a red box between min_threshold and max threshold
    ax.axhspan(min_threshold, max_threshold, color="red", alpha=0.25, label="Analysis Region")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, max(t["range"]))
    ax.set_yticks(t["range"])
    ax.legend(loc="upper right", fontsize="small")

    name = t["name"]

    ax.set_title(rf"$\mathit{{{name}}}$ in {t['material']}")

    # fig.savefig(
    #     f"projects/Dissertation/proposal/figures/nspa/{t['target']}_{t['material']}_nspa.pdf",
    #     dpi=300,
    # )
# %%
