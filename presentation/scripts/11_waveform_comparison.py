# %%

import matplotlib.pyplot as plt
import pandas as pd
from spidb import spidb

from aspids_tools import normalization

# %%
db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")

targets = [
    {
        "target": "Confused flour beetle",
        "name": "T.\ confusum",
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
        "name": "C.\ maculatus",
        "material": "Wheat Groats",
        "channel": 0,
        "record": 329,
        "amplitude": 1000,
        "amplitude2": 1200,
    },
    {
        "target": "Noise",
        "name": "Noise 60 dBA",
        "material": "60 dBA",
        "channel": 0,
        "record": 460,
        "amplitude": 250,
        "amplitude2": 10,
    },
    {
        "target": "Noise",
        "name": "Noise 90 dBA",
        "material": "90 dBA",
        "channel": 3,
        "record": 463,
        "amplitude": 2800,
        "amplitude2": 100,
    },
]

# %%
# alpha = ["a", "b", "c", "d"]
low = 500
high = 6000

fig, axs = plt.subplots(figsize=(11.71, 4.24), nrows=2, ncols=2, sharex=True)

for i, t in enumerate(targets):
    
    ax = axs[i // 2, i % 2]  # Get the appropriate subplot

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

    # fig, ax = plt.subplots(figsize=(3.2, 1.81))
    ax.plot(audio.data.seconds, audio.data.signal, label=f"NSPA {nspa:.0f} dB")
    if i in [2, 3]:
        ax.set_xlabel("Time [s]")

    ax.set_xlim(0, 60)
    ax.set_ylim(0, t["amplitude"])
    ax.set_yticks([0, round(t["amplitude"] / 2), t["amplitude"]])

    # remove marker lines from legend
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)
    if t["target"] == "Noise":
        ax.set_title(f"{t['name']}")
    else:
        ax.set_title(rf"$\mathit{{{t['name']}}}$ in {t['material']}")
fig.supylabel("Normalized Amplitude")
#%%