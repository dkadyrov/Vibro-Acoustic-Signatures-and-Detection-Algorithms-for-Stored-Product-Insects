# %%
from spidb import spidb, normalization, visualizer, detection
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dankpy import acoustics, color
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal

plt.style.use("dankpy.styles.stevens_presentation")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")
#%%
# events = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49] 
records = [
    {"start": "2023-12-18 19:57:00", "end": "2023-12-18 19:57:30", "measured": 60.0},
    {"start": "2023-12-18 19:57:30", "end": "2023-12-18 19:58:00", "measured": 60.0},
    {"start": "2023-12-18 19:58:00", "end": "2023-12-18 19:58:30", "measured": 70.0},
    {"start": "2023-12-18 19:58:30", "end": "2023-12-18 19:59:00", "measured": 70.0},
    # {"start": "2023-12-18 19:59:00", "end": "2023-12-18 20:59:30", "measured": 80.0},
    {"start": "2023-12-18 19:59:30", "end": "2023-12-18 20:00:00", "measured": 80.0},
    {"start": "2023-12-18 20:00:00", "end": "2023-12-18 20:00:30", "measured": 90.0},
    {"start": "2023-12-18 20:00:30", "end": "2023-12-18 20:01:00", "measured": 90.0},
    # {"start": "2023-12-18 20:01:30", "end": "2023-12-18 20:02:00", "measured": 100.0},
    # {"start": "2023-12-18 20:08:00", "end": "2023-12-18 20:08:30", "measured": 70.0},
    {"start": "2023-12-18 20:08:30", "end": "2023-12-18 20:09:00", "measured": 70.0},
    {"start": "2023-12-18 20:09:30", "end": "2023-12-18 20:10:00", "measured": 80.0},
    {"start": "2023-12-18 20:10:20", "end": "2023-12-18 20:11:00", "measured": 90.0},
    {"start": "2023-12-18 20:11:00", "end": "2023-12-18 20:11:30", "measured": 100.0},
    {"start": "2023-12-18 20:11:30", "end": "2023-12-18 20:12:00", "measured": 100.0},
]

data = pd.DataFrame(records)
data["start"] = pd.to_datetime(data["start"])
data["end"] = pd.to_datetime(data["end"])
sensor = db.session.get(spidb.Sensor, 1)

for i, event in data.iterrows(): 
    start = event["start"]
    end = event["end"]

    audio = db.get_audio(start, end, sensor=sensor, channel_number=7)

    spl = acoustics.calculate_spl(audio.data.signal, audio.sample_rate)
    spl = normalization.spl_coefficient(spl)

    nspa_1 = []
    nspa_2 = []   
    nspa_3 = []
    for j, channel in enumerate(sensor.channels[:4]):
        audio = db.get_audio(start, end, sensor=sensor, channel_number=channel.number)

        nspa = normalization.calculate_nspa(audio, filter="bandpass", low=500, high=6000, channel=channel, db=db, normalize="noise")
        nspa_1.append(nspa)

        nspa = normalization.calculate_nspa(audio, filter="bandpass", low=1565, high=6000, channel=channel, db=db, normalize="noise")
        nspa_2.append(nspa)

        nspa = normalization.calculate_nspa(audio, filter="bandpass", low=1565, high=8000, channel=channel, db=db, normalize="noise")
        nspa_3.append(nspa)

    # d = {
    #     "spl": spl,
    #     "NSPA (500-6000)": max(nspa_1),
    #     "NSPA (1565-6000)": max(nspa_2),
    # }

    data.at[i, "spl"] = spl
    data.at[i, "NSPA (500-6000)"] = max(nspa_1)
    data.at[i, "NSPA (1565-6000)"] = max(nspa_2)
    data.at[i, "NSPA (1565-8000)"] = max(nspa_3)

data = pd.DataFrame(data)
#%%
fig, ax = plt.subplots()
ax.scatter(data["spl"], data["NSPA (500-6000)"], label="500-6000 Hz", color=color.colors[0], marker="o")
ax.scatter(data["spl"], data["NSPA (1565-6000)"], label="1565-6000 Hz", color=color.colors[1], marker="s")
ax.scatter(data["spl"], data["NSPA (1565-8000)"], label="1565-8000 Hz", color=color.colors[2], marker="^")
# %%
data.groupby("spl").mean()
# %%
# round to the nearest 10
data["spl_round"] = data["spl"].round(-1)
# %%
data.groupby("spl_round").mean()

# %%
# Create a linear regression model for 1565-6000
X = data[data["spl"]<94]["spl"].values.reshape(-1, 1)
y = data[data["spl"]<94]["NSPA (1565-6000)"].values

model = LinearRegression()
model.fit(X, y)

ax.plot(np.linspace(60, 110, 100), model.predict(np.linspace(60, 110, 100).reshape(-1, 1)), color=color.colors[1], linestyle="--", label="1565-6000 Hz Fit")

# get model parameters
print(f"1565-6000 Hz: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

# %%
X = data[data["spl"]<94]["spl"].values.reshape(-1, 1)
y = data[data["spl"]<94]["NSPA (500-6000)"].values

model = LinearRegression()
model.fit(X, y)

ax.plot(np.linspace(60, 110, 100), model.predict(np.linspace(60, 110, 100).reshape(-1, 1)), color=color.colors[0], linestyle="--", label="500-6000 Hz Fit")
# %%
