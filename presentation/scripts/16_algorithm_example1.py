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

plt.style.use("dankpy.styles.stevens_presentation")
# %%
db = spidb.Database(r"data/spi.db")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

noise_record = db.session.get(spidb.Record, 464)

noise_external = db.get_audio(
    sensor=noise_record.sensor,
    start=noise_record.start
    + timedelta(seconds=40),  # - timedelta(seconds=20),  # + timedelta(seconds=41),
    end=noise_record.end,  # - timedelta(seconds=20),  # - timedelta(seconds=7),
    channel_number=7,
)
noise_external = processing.process_signal(
    db, noise_external, channel=noise_record.sensor.channels[7]
)

noise_internal = db.get_audio(
    sensor=noise_record.sensor,
    start=noise_record.start
    + timedelta(seconds=40),  # - timedelta(seconds=20),  # + timedelta(seconds=41),
    end=noise_record.end,  # - timedelta(seconds=20),  # - timedelta(seconds=7),
    channel_number=0,
)

insect_record = db.session.get(spidb.Record, 1942)
insect = db.get_audio(
    sensor=insect_record.sensor,
    start=insect_record.start + timedelta(seconds=2),
    end=insect_record.start
    + timedelta(seconds=22),  # record.start + timedelta(seconds=47),
    channel_number=0,
)

insect = processing.process_signal(db, insect, channel=insect_record.sensor.channels[0])


combined = deepcopy(insect)

T = 21.6
K = 30.54
A = 10 ** (T / 20) * 10 ** (6 / 20)
KT = 10 ** (T / 20) * 10 ** (K / 20) / 10 ** (6 / 20)
# KT = A / (2 * 10 ** (K / 20))

peaks, _ = processing.find_peaks(insect.data.signal, threshold=A, min_distance=1000)

# insect_nspa = np.sqrt(np.mean(insect.data.signal[insect.data.signal >= A] ** 2))
# insect_nspa = 20 * np.log10(insect_nspa)

insect_nspa = normalization.nspa(insect.data.signal)

fig, ax = plt.subplots(figsize=(3.61, 4.21))
ax.plot(
    insect.data.seconds,
    insect.data.signal,
    label=f"Ch. 0, NSPA {insect_nspa:.0f} dB",
)
ax.scatter(
    insect.data.seconds[peaks],
    insect.data.signal[peaks],
    color="red",
    label=f"{len(peaks)} Impulses",
)
ax.set_xlim(0, 20)
ax.set_ylim(0, 120)
ax.set_yticks([0, 60, 120])
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)
ax.set_ylabel("Normalized Amplitude")
ax.set_xlabel("Time [s]")
ax.set_title("$\\mathit{T. confusum}$ in Rice")

#%%

noise_internal = processing.process_signal(
    db, noise_internal, channel=noise_record.sensor.channels[0]
)

# noise_internal_nspa = normalization.nspa(noise_internal.data.signal)
# noise_external_nspa = normalization.nspa(noise_external.data.signal)

combined.data.loc[:, "signal"] = combined.data["signal"] + noise_internal.data["signal"]
combined.audio += noise_internal.audio

combined_nspa = normalization.nspa(combined.data.signal)
# combined = processing.process_signal(
# db, combined, channel=insect_record.sensor.channels[0]
# )

# A = 0.5 * np.max(combined.data.signal)
# combined_nspa = np.sqrt(np.mean(combined.data.signal[combined.data.signal >= A] ** 2))
# combined_nspa = 20 * np.log10(combined_nspa)
# combined_nspa = norma

peaks, _ = processing.find_peaks(combined.data.signal, threshold=A, min_distance=1000)

fig, ax = plt.subplots(figsize=(3.61, 4.21))
ax.plot(
    combined.data.seconds,
    combined.data.signal,
    label=f"Ch. 0, NSPA {combined_nspa:.0f} dB",
)
ax.scatter(
    combined.data.seconds[peaks],
    combined.data.signal[peaks],
    color="red",
    label=f"{len(peaks)} Impulses",
)
ax.set_xlim(0, 20)
ax.set_ylim(0, 120)
ax.set_yticks([0, 60, 120])
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)
ax.set_ylabel("Normalized Amplitude")
ax.set_xlabel("Time [s]")
ax.set_title("Insect and Noise Signal")

noise_peaks, _ = processing.find_peaks(
    noise_external.data.signal, threshold=KT, min_distance=100
)

# remove these peaks from the combined signal
for peak in noise_peaks:
    # Convert peak index to time
    peak_time = noise_external.data.seconds[peak]

    # Find corresponding index in combined signal
    time_diff = np.abs(combined.data.seconds - peak_time)
    combined_peak_idx = np.argmin(time_diff)

    # Zero out the region around the peak without chained assignment
    start_idx = max(0, combined_peak_idx - 500)
    end_idx = min(len(combined.data.signal), combined_peak_idx + 500)
    combined.data.loc[start_idx:end_idx, "signal"] = 0

# A = 10 ** (T / 20) * 10 ** (6 / 20)
# cleaned_nspa = np.sqrt(np.mean(combined.data.signal[combined.data.signal >= A] ** 2))
# cleaned_nspa = 20 * np.log10(cleaned_nspa)

cleaned_nspa = normalization.nspa(combined.data.signal)

peaks, _ = processing.find_peaks(combined.data.signal, threshold=A, min_distance=1000)

fig, ax = plt.subplots(figsize=(3.61, 4.21))
ax.plot(
    combined.data.seconds,
    combined.data.signal,
    label=f"Ch. 0, NSPA {cleaned_nspa:.0f} dB",
)
ax.scatter(
    combined.data.seconds[peaks],
    combined.data.signal[peaks],
    color="red",
    label=f"{len(peaks)} Impulses",
)
ax.set_xlim(0, 20)
ax.set_xlabel("Time [s]")
ax.set_ylim(0, 120)
ax.set_yticks([0, 60, 120])
ax.set_ylabel("Normalized Amplitude")
ax.legend(loc="upper right", handlelength=0, handletextpad=0, markerscale=0)
ax.set_title("After Algorithm")
# %%
