# %%
from spidb import spidb, normalization, visualizer, detection
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dankpy import acoustics, color
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.signal import lfilter, bilinear

plt.style.use("dankpy.styles.latex")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")
# %%
events = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49]
events = [db.session.get(spidb.Event, e) for e in events]
events = pd.DataFrame([e.__dict__ for e in events])
events["type"] = events["description"].str.split("-").str[0].str.strip()
#%%
records = [
    {"start": "2023-12-18 19:57:00", "end": "2023-12-18 19:57:30", "measured": 60.0},
    {"start": "2023-12-18 19:57:30", "end": "2023-12-18 19:58:00", "measured": 60.0},
    {"start": "2023-12-18 19:58:00", "end": "2023-12-18 19:58:30", "measured": 70.0},
    {"start": "2023-12-18 19:58:30", "end": "2023-12-18 19:59:00", "measured": 70.0},
    # {"start": "2023-12-18 19:59:00", "end": "2023-12-18 20:59:30", "measured": 80.0},
    {"start": "2023-12-18 19:59:30", "end": "2023-12-18 20:00:00", "measured": 80.0},
    {"start": "2023-12-18 20:00:00", "end": "2023-12-18 20:00:30", "measured": 90.0},
    {"start": "2023-12-18 20:00:30", "end": "2023-12-18 20:01:00", "measured": 90.0},
    {"start": "2023-12-18 20:01:30", "end": "2023-12-18 20:02:00", "measured": 100.0},
    {"start": "2023-12-18 20:08:00", "end": "2023-12-18 20:08:30", "measured": 70.0},
    {"start": "2023-12-18 20:08:30", "end": "2023-12-18 20:09:00", "measured": 70.0},
    {"start": "2023-12-18 20:09:30", "end": "2023-12-18 20:10:00", "measured": 80.0},
    {"start": "2023-12-18 20:10:20", "end": "2023-12-18 20:11:00", "measured": 90.0},
    {"start": "2023-12-18 20:11:00", "end": "2023-12-18 20:11:30", "measured": 100.0},
    {"start": "2023-12-18 20:11:30", "end": "2023-12-18 20:12:00", "measured": 100.0},
]

records = pd.DataFrame(records)
records["start"] = pd.to_datetime(records["start"])
records["end"] = pd.to_datetime(records["end"])
# have record["description"] match events["description"] based on the time overlap
records["type"] = ""
for i, row in records.iterrows():
    overlap = events[
        (events["start"] < row["end"]) & (events["end"] > row["start"])
    ]
    if len(overlap) > 0:
        records.at[i, "type"] = overlap.iloc[0]["type"]
#%%

sensor = db.session.get(spidb.Sensor, 1)

# %%
rmss = []
logs = []

spls = []
nsels = []

# fig, ax = plt.subplots()
# for e, group in events.groupby("measured"):
    # for i, row in group.iterrows():
for i, row in records.iterrows():
    audio = db.get_audio(row["start"], row["end"], sensor=sensor, channel_number=7)

    # fig, ax = audio.plot_spectrogram(window="hann", window_size=1024, nfft=1024, noverlap=512, nperseg=1024, zmax=-80, zmin=-140, time_format="seconds", cmap="jet", showscale="right")
    # ax.set_ylim(0, 8000)
    # ax.set_title(f"{row['type']} - {row['measured']} dBA")

    # f, p = signal.welch(
    #     audio.data.signal,
    #     fs=audio.sample_rate,
    #     nperseg=1024,
    #     window="blackmanharris",
    #     average="mean",
    # )

    # p = 10 * np.log10(p)

    # fig, ax = plt.subplots()
    # ax.plot(f, p)
    # ax.set_xlim(0, 12000)
    # ax.set_ylim(-150, -50)
    # ax.set_xlabel("Frequency [Hz]")
    # ax.set_ylabel("Spectral Power [dB]", fontsize=7)
    # ax.set_title(f"{row['type']} - {row['measured']} dBA")

    spl = acoustics.calculate_spl_dba(audio.data.signal, audio.sample_rate)
    spls.append(spl)

    rms = np.sqrt(np.mean(audio.data.signal**2))
    log = 20 * np.log10(rms)

    rmss.append(rms)
    logs.append(log)

        # f, p = signal.welch(
        #     audio.data.signal,
        #     fs=audio.sample_rate,
        #     nperseg=1024,
        #     window="blackmanharris",
        #     average="mean",
        # )

        # p = 10 * np.log10(p)

        # ax.plot(f, p, alpha=0.5)

        # spl = 10 * np.log10(np.sum(10 ** (p / 10)))

        # spls.append(spl)

        # nsel = normalization.calculate_nsel(
        #     audio,
        #     filter="bandpass",
        #     low=2000,
        #     high=10000,
        #     channel=sensor.channels[-1],
        #     db=db,
        # )
        # nsels.append(nsel)
records["rms"] = rmss
records["db"] = logs
records["spl"] = spls
records["difference"] = records["measured"] - records["db"]
# %%
fig, ax = plt.subplots()
for i, group in records.groupby("type"):
    ax.scatter(group["db"], group["difference"], label=f"{i}")
ax.set_xlabel("SPL Meter Measurement [dBA]")
ax.set_ylabel("Difference [dB]")
ax.legend(loc="upper left")
ax.set_ylim(80, 160)
ax.set_yticks([80, 120, 160])
# %%
fig, ax = plt.subplots()
for i, group in records.groupby("type"):
    ax.scatter(group["measured"], group["db"], label=f"{i}")
ax.set_xlabel("SPL Meter Measurement [dBA]")
ax.set_ylabel("SPL [dBFS]")
ax.legend(loc="upper left")
# %%
# curve fit the difference using linear regression
model = LinearRegression()
X = records["db"].values.reshape(-1, 1)
y = records["measured"].values
model.fit(X, y)
print(f"Slope: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")
# %%
model = LinearRegression()
X = records["difference"].values.reshape(-1, 1)
y = records["measured"].values
model.fit(X, y)
print(f"Slope: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")

#%%
def correction(x):
    # difference = x * 2.2 + 176.41
    # measured = 1.39*difference - 73.85
    measured = 1.39*x - 73.85

    return measured

#%%
fig, ax = plt.subplots()
for i, group in records.groupby("type"):
    ax.scatter(group["measured"], group["db"], label=f"{i}")
ax.set_xlabel("SPL Meter Measurement [dBA]")
ax.set_ylabel("Channel 7 Measurement [dB]")
ax.legend(loc="upper left")
# ax.set_ylim(80, 160)
# ax.set_yticks([80, 120, 160])


#%%
fig, ax = plt.subplots()
for i, group in records.groupby("type"):
    ax.scatter(group["measured"], group["db"], label=f"{i}")
ax.plot(np.linspace(50, 110, 100), correction(np.linspace(-40, 20, 100)), color="gray", linestyle="--", label="Linear Fit")
ax.set_xlabel("SPL Meter Measurement [dBA]")
ax.set_ylabel("Difference [dB]")
ax.legend(loc="upper left", ncols=3)
# ax.set_ylim(80, 160)
ax.set_xlim(50, 110)
# ax.set_yticks([80, 120, 160])
# %%
