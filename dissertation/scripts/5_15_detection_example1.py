# %%
from copy import deepcopy
from datetime import timedelta

import cblind as cb  # type: ignore  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.signal import welch
from spidb import spidb

from aspids_tools import normalization, processing, visualizer

plt.style.use("dankpy.styles.latex")

db = spidb.Database(r"data/spi.db")

record = db.session.get(spidb.Record, 464)

audios = db.get_audios(
    sensor=record.sensor,
    start=record.start,
    end=record.end,
    channels=[0, 1, 2, 3, 7],
)

audios = [
    processing.process_signal(
        db,
        audio,
        record.sensor.channels[audio.channel_number],
        fade_time=1,
        low=1565,
        high=6000,
    )
    for audio in audios
]

# insect = db.session.get(spidb.Record, 1121)
insect = db.session.get(spidb.Record, 1095)
insect_audios = db.get_audios(
    sensor=insect.sensor,
    start=insect.start + timedelta(seconds=40),
    end=insect.end + timedelta(seconds=40),
    channels=[0, 1, 2, 3],
)

insect_audios = [
    processing.process_signal(
        db,
        a,
        channel=insect.sensor.channels[a.channel_number],
        fade_time=1,
        low=1565,
        high=6000,
    )
    for a in insect_audios
]

for i in range(4):
    audios[i].data.signal += insect_audios[i].data.signal
    audios[i].audio += insect_audios[i].audio

T = 21.6
K_mean = 30.54
KT = 10 ** (T / 20) * 10 ** (K_mean / 20) / 10 ** (6 / 20)
A = 10 ** (21.6 / 20) * 10 ** (6 / 20)

noise_peaks, _ = processing.find_peaks(
    audios[-1].data.signal, threshold=KT, min_distance=250
)

window_size = int(11025.0 // 2)

for a in audios[:-1]:
    for peak in noise_peaks:
        peak_time = audios[-1].data.seconds[peak]
        time_diff = np.abs(a.data.seconds - peak_time)
        combined_peak_idx = np.argmin(time_diff)
        start_idx = max(0, combined_peak_idx - window_size)
        end_idx = min(len(a.data.signal), combined_peak_idx + window_size)
        a.data.signal.values[start_idx:end_idx] = 0

# nspas = [normalization.nspa(a.data.signal) for a in audios]
nspas = [
    20
    * np.log10(
        np.sqrt(np.mean(a.data.signal[a.data.signal >= (A if i < 4 else KT)] ** 2))
    )
    for i, a in enumerate(audios)
]
# replace nspas that have nan with "None"
nspas = [round(nspa, 2) if not np.isnan(nspa) else None for nspa in nspas]

# labels = [f"Ch. {a.channel_number}, NSPA {nspa} dB" for a, nspa in zip(audios, nspas)]
labels = [f"Ch. {a.channel_number}" for a in audios]
# add NSPA only if not None
labels = [
    f"{label}, NSPA {nspa} dB" if nspa is not None else label
    for label, nspa in zip(labels, nspas)
]

# %%
fig, ax = visualizer.waveform_display(audios, labels=labels, time_format="seconds")
for i, a in enumerate(ax):
    if i < 4:
        # if i < len(ax) - 3:
        peaks, _ = processing.find_peaks(
            audios[i].data.signal,
            threshold=A,
            min_distance=250,
        )
        a.set_ylim(0, 150)
        a.set_yticks([0, 75, 150])
    else:
        peaks = noise_peaks

    num_peaks = len(peaks)
    if i < 4:
        result = "Insect" if num_peaks > 5 else "Clean"
        color = "red"
    else:
        result = "Noise" if nspas[i] > 70 else "Silence"
        color = "blue"
    # make the label color red if result is not "Clean"
    a.scatter(
        audios[i].data.seconds[peaks],
        audios[i].data.signal[peaks],
        color=color,
        alpha=0.5,
        label=f"{len(peaks)} Impulses",
    )
    a.legend(
        loc="upper right", handlelength=0, handletextpad=0, markerscale=0, fontsize=8
    )

    # draw a full-height colored bar to the right of each axis and rotate the label 180 degrees
    color_map = {
        "Clean": "tab:green",
        "Insect": "tab:red",
        "Noise": "tab:blue",
        "Silent": "lightgrey",
    }

    color = color_map.get(result, "lightgrey")

    rect = mpatches.Rectangle(
        (1.025, 0),
        0.05,
        1,
        transform=a.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        clip_on=False,
        zorder=2,
    )
    a.add_patch(rect)

    rect = mpatches.Rectangle(
        (1.025, 0.75),
        0.05,
        0.25,
        transform=a.transAxes,
        facecolor=color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        clip_on=False,
        zorder=2,
    )
    a.add_patch(rect)

    a.text(
        1.025 + 0.02125,
        0.60,
        f"{result}",
        transform=a.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="center",
        rotation=-90,
        color="black",
        clip_on=False,
        zorder=3,
    )

fig.set_size_inches(6, 5)
fig.savefig(
    # fix labels to accomodate for a None
    r"projects\Dissertation\dissertation\figures\\5_15_waveforms_with_insect.pdf",
    dpi=300,
)
# %%
