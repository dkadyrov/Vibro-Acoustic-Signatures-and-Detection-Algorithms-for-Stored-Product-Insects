# %%

import matplotlib.pyplot as plt
import pandas as pd
from spidb import spidb

from aspids_tools import normalization

# %%
db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

targets = [
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "channel": 2,
        "record": 1688,
        "amplitude": 50,
        "amplitude2": 50,
    },
    # {
    #     "target": "Bean beetle",
    #     "material": "Flour",
    #     "channel": 3,
    #     "record": 19,
    #     "amplitude": 50,
    #     "amplitude2": 50,
    # },
    # {
    #     "target": "Bean beetle",
    #     "material": "Oatmeal",
    #     "channel": 0,
    #     "record": 50,
    #     "amplitude": 75,
    #     "amplitude2": 20,
    # },
    {
        "target": "Bean beetle",
        "material": "Wheat Groats",
        "channel": 0,
        "record": 329,
        "amplitude": 1000,
        "amplitude2": 1200,
    },
    {
        "target": "Noise",
        "material": "60 dBA",
        "channel": 0,
        "record": 460,
        "amplitude": 250,
        "amplitude2": 10,
    },
    {
        "target": "Noise",
        "material": "80 dBA",
        "channel": 3,
        "record": 463,
        "amplitude": 2800,
        "amplitude2": 100,
    },
]

# %%
alpha = ["a", "b", "c", "d"]
low = 1565
high = 6000

for i, t in enumerate(targets):
    record = db.session.get(spidb.Record, t["record"])

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    channel = record.sensor.channels[t["channel"]]

    nspa = normalization.calculate_nspa(
        audio,
        filter="bandpass",
        low=low,
        high=high,
        normalize="noise",
        channel=channel,
        db=db,
    )

    audio.fade_in(1, overwrite=True)
    audio.fade_out(1, overwrite=True)

    audio.bandpass_filter(low, high, 10, overwrite=True)

    audio.envelope(overwrite=True)

    audio = normalization.noise_normalize(
        db,
        audio,
        channel=channel,
        filter="bandpass",
        low=low,
        high=high,
        coefficient="set",
    )

    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.plot(audio.data.seconds, audio.data.signal, label=f"NSPA {nspa:.2f} dB")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized\nAmplitude")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, t["amplitude2"])
    ax.set_yticks([0, round(t["amplitude2"] / 2), t["amplitude2"]])

    # remove marker lines from legend
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)

    fig.savefig(
        f"projects\\Dissertation\\dissertation\\figures\\5_8{alpha[i]}_{t['target']}_{t['material']}_waveform_normalized.pdf",
        dpi=300,
    )
# %%
