# %%
from copy import deepcopy
from datetime import timedelta

import cblind as cb  # type: ignore  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sonicdb import audio
from spidb import spidb

from aspids_tools import normalization, processing

plt.style.use("dankpy.styles.latex")

db = spidb.Database(r"data/spi.db")

filepath = r"data\ESC-50\audio\1-19898-A-41.wav"

noise_external = audio.Audio(filepath)

scale_record = db.session.get(spidb.Record, 463)

scale_audio = db.get_audio(
    sensor=scale_record.sensor,
    start=scale_record.start,
    end=scale_record.end,
    channel_number=7,
)
scale_audio.resample(22050)
scale_audio = processing.process_signal(
    db,
    scale_audio,
    channel=scale_record.sensor.channels[7],
    fade_time=1,
    low=1565,
    high=6000,
)

noise_external = processing.process_signal(
    db,
    noise_external,
    channel=scale_record.sensor.channels[7],
    fade_time=0.25,
    low=1565,
    high=6000,
)

rms_scale = np.sqrt(np.mean(scale_audio.data.signal**2))
rms_noise_external = np.sqrt(np.mean(noise_external.data.signal**2))

ratio = rms_scale / rms_noise_external

noise_external.data.loc[:, "signal"] = noise_external.data["signal"] * ratio
noise_external.audio *= ratio

T = 21.6
K = 30.54
KT = 10 ** (T / 20) * 10 ** (K / 20) / 10 ** (6 / 20)

noise_peaks, _ = processing.find_peaks(
    noise_external.data.signal, threshold=KT, min_distance=250
)

fig, ax = plt.subplots()
ax.plot(
    noise_external.data.seconds,
    noise_external.data.signal,
    label=f"Ch. 7, NSPA {normalization.nspa(noise_external.data.signal):.2f} dB",
)
ax.scatter(
    noise_external.data.seconds[noise_peaks],
    noise_external.data.signal[noise_peaks],
    color="red",
    s=20,
    label=f"{len(noise_peaks)} Peaks",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized\nAmplitude")
ax.set_ylim(0, None)
ax.set_xlim(0, 5)
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)

# %%

noise_internal = audio.Audio(filepath)

noise_internal = processing.process_signal(
    db,
    noise_internal,
    channel=scale_record.sensor.channels[0],
    fade_time=0.25,
    low=1565,
    high=6000,
)

noise_internal.data.loc[:, "signal"] = noise_internal.data["signal"] * (
    ratio / 10 ** (30.54 / 20)
)
noise_internal.audio *= ratio / 10 ** (30.54 / 20)

insect = db.session.get(spidb.Record, 941)
insect = db.get_audio(
    sensor=insect.sensor,
    start=insect.start + timedelta(seconds=40.5),
    end=insect.start + timedelta(seconds=45.5),
    channel_number=0,
)
insect = processing.process_signal(
    db,
    insect,
    channel=scale_record.sensor.channels[0],
    fade_time=0.25,
    low=1565,
    high=6000,
)
insect.resample(22050)

A = 10 ** (21.6 / 20) * 10 ** (6 / 20)

peaks, _ = processing.find_peaks(insect.data.signal, threshold=A, min_distance=500)

insect_nspa = np.sqrt(np.mean(insect.data.signal[insect.data.signal >= A] ** 2))
insect_nspa = 20 * np.log10(insect_nspa)

fig, ax = plt.subplots()
ax.plot(
    insect.data.seconds,
    insect.data.signal,
    label=f"Ch. 0, NSPA {insect_nspa:.2f} dB",
)
ax.scatter(
    insect.data.seconds[peaks],
    insect.data.signal[peaks],
    color="red",
    s=20,
    label=f"{len(peaks)} Peaks",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized\nAmplitude")
ax.set_ylim(0, 75)
ax.set_xlim(0, 5)
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\5_12a_external_noise_algorithm_chainsaw_insect.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
combined = deepcopy(insect)
combined.data.loc[:, "signal"] = combined.data["signal"] + noise_internal.data["signal"]
combined.audio += noise_internal.audio

peaks, _ = processing.find_peaks(combined.data.signal, threshold=A, min_distance=250)

combined_nspa = np.sqrt(np.mean(combined.data.signal[combined.data.signal >= A] ** 2))
combined_nspa = 20 * np.log10(combined_nspa)


fig, ax = plt.subplots()
ax.plot(
    combined.data.seconds,
    combined.data.signal,
    label=f"Ch. 0, NSPA {combined_nspa:.2f} dB",
)
ax.scatter(
    combined.data.seconds[peaks],
    combined.data.signal[peaks],
    color="red",
    s=20,
    label=f"{len(peaks)} Peaks",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized\nAmplitude")
ax.set_ylim(0, 75)
ax.set_xlim(0, 5)
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\5_12b_external_noise_algorithm_chainsaw_combined.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
for peak in noise_peaks:
    # Convert peak index to time
    peak_time = noise_external.data.seconds[peak]

    # Find corresponding index in combined signal
    time_diff = np.abs(combined.data.seconds - peak_time)
    combined_peak_idx = np.argmin(time_diff)

    # Zero out the region around the peak without chained assignment
    start_idx = max(0, combined_peak_idx - 250)
    end_idx = min(len(combined.data.signal), combined_peak_idx + 250)
    combined.data.loc[start_idx:end_idx, "signal"] = 0

peaks, _ = processing.find_peaks(combined.data.signal, threshold=A, min_distance=250)

cleaned_nspa = np.sqrt(np.mean(combined.data.signal[combined.data.signal >= A] ** 2))
cleaned_nspa = 20 * np.log10(cleaned_nspa)

fig, ax = plt.subplots()
ax.plot(
    combined.data.seconds,
    combined.data.signal,
    label=f"Ch. 0, NSPA {cleaned_nspa:.2f} dB",
)
ax.scatter(
    combined.data.seconds[peaks],
    combined.data.signal[peaks],
    color="red",
    s=20,
    label=f"{len(peaks)} Peaks",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized\nAmplitude")
ax.set_ylim(0, 75)
ax.set_xlim(0, 5)
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\5_12c_external_noise_algorithm_chainsaw_combined_cleaned.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
