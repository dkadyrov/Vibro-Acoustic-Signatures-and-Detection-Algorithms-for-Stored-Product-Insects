# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from spidb import spidb
from aspids_tools import normalization

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

targets = [
    {
        "target": "Mealworm",
        "name": "T.\ molitor",
        "material": "Wheat Groats",
        "test": 93,
        "iteration": 12,
        "channel": 2,
        "record": 1007,
        "range": [0, 4000, 8000]
    },
]

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

    audio.envelope(overwrite=True)
    audio = normalization.noise_normalize(
        db,
        audio,
        channel=channel,
        filter="bandpass",
        low=500,
        high=6000,
        coefficient="set",
    )

    min_threshold = 0.5 * np.max(audio.data.signal)
    max_threshold = 0.9 * np.max(audio.data.signal)

    cutoff = audio.data.signal[audio.data.signal >= min_threshold]
    cutoff = cutoff[cutoff <= max_threshold]

    rms = np.sqrt(np.mean(cutoff**2))
    amp = 20 * np.log10(rms)

    RMS = np.sqrt(np.mean(audio.data.signal**2))
    AMP = 20 * np.log10(RMS)

    fig, ax = plt.subplots(figsize=(5.67, 4.21))
    ax.plot(audio.data.seconds, audio.data.signal)

    ax.axhline(
        RMS, color="blue", linestyle="--", label=f"RMS ({AMP:.0f} dB)", zorder=10
    )

    # make a dashed red line at the RMS value, round it to 2 decimal places
    ax.axhline(
        rms, color="red", linestyle="--", label=f"NSPA ({amp:.0f} dB)", zorder=10
    )

    # make a red box between min_threshold and max threshold
    ax.axhspan(
        min_threshold, max_threshold, color="red", alpha=0.25, label="NSPA Analysis Region"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, max(t["range"]))
    ax.set_yticks(t["range"])
    ax.legend(loc="upper right", ncols=1)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    name = t["name"]

    ax.set_title(rf"$\it{{{name}}}$ larva in {t['material']}")

# %%
