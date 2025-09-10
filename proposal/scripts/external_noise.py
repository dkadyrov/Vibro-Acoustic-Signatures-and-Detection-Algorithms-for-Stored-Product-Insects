# %%
from spidb import spidb, normalization, visualizer, detection
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dankpy import acoustics, color
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal

plt.style.use("dankpy.styles.latex")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")
#%%
# events = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49] 
events = [41, 42, 43, 44]
data = [] 

for i, event in enumerate(events): 
    event = db.session.get(spidb.Event, event)

    # start = event.start + timedelta(seconds=iterat[i]*10) #+ timedelta(seconds=30)
    # end = start + timedelta(seconds=10)
    start = event.start
    end = start + timedelta(seconds=60)

    audio = db.get_audio(start, end, sensor=event.sensor, channel_number=7)
    audio.fade_in(fade_time=1, overwrite=True)
    audio.fade_out(fade_time=1, overwrite=True)

    spl = acoustics.calculate_spl_dba(audio.data.signal, audio.sample_rate)

    nspas = []
    nsels = []

    fig, ax = plt.subplots()
    for i, channel in enumerate(event.sensor.channels[4:]):
        audio = db.get_audio(start, end, sensor=event.sensor, channel_number=channel.number)
        f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")

        p = 10 * np.log10(p)
        snr = p - 10*np.log10(noise[f"{channel.number}"])

        ax.plot(f, snr, label=f"Ch. {channel.number}", color=color.colors[i])

    ax.set_xlim(0, 8000)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 35, 70])
    ax.set_ylabel("SNR [dB]")
    ax.legend(loc="upper right", ncols=4)
    fig.savefig(f"projects/Dissertation/proposal/figures/external_noise/snr_{event.id}.pdf", dpi=300)

    data.append({
        "event": event.id,
        "type": event.description.split("-")[0],
        "actual": int(event.description.split("-")[1].strip().replace("dB", "").strip()),
        "spl": spl,
    })
    #%%
    spl = normalization.spl_coefficient(spl)

    fig, ax = visualizer.spectra_display(db, start=start, end=end, sensor=event.sensor, section="external")
    ax.legend(loc="upper right", ncols=4)

    ax.set_xlim(0, 8000)
    ax.set_ylim(-125, -25)
    ax.set_yticks([-125, -75, -25])
    fig.savefig(rf"projects/Dissertation/proposal/figures/external_noise/external_noise_{spl:.2f}.pdf", dpi=300)




    #%%

data = pd.DataFrame(data)
data["ratio"] = data["spl"] / data["actual"]
data["diff"] = data["actual"] - data["spl"]

data = data.sort_values(by="actual")

#%%
average_df = pd.DataFrame()
average_df["actual"] = data["actual"].unique()
average_df["spl"] = data.groupby(by="actual")["spl"].mean().values
#%%


model = LinearRegression(fit_intercept=True)
X = data["spl"].values.reshape(-1, 1)
y_actual = data["actual"].values.reshape(-1, 1)
model.fit(X, y_actual)
x = np.linspace(20, 80, 100)
y = model.predict(x.reshape(-1, 1))

r2 = model.score(X, y_actual)
print(f"R^2 score: {r2:.3f}")  # Add this line to output the accuracy

m = model.coef_[0][0]
b = model.intercept_[0]

def calculate_actual(spl): 
    return m * spl + b

data["model"] = data.spl.apply(calculate_actual)


#%%
fig, ax = plt.subplots()
j=0
for i, t in data.groupby(by="type"): 
    ax.scatter(t["spl"], t["actual"], marker="o", label="Noise")
    j += 1
# ax.scatter(average_df["actual"], average_df["spl"], marker="*", label="Average", color="black")
ax.plot(x, y, label="Fit", color="black", linestyle="--")
ax.legend(loc="upper right", ncols=4)
ax.set_xlabel("SPL Ch. 7 [dBA]")
ax.set_ylabel("SPL Meter [dBA]")
ax.set_ylim(0, 150)
ax.set_yticks([0, 75, 150])
ax.set_xlim(20, 80)
ax.set_xticks([20, 40, 60, 80])
fig.savefig(r"projects/Dissertation/proposal/figures/external_noise/ch7_spl_graph.pdf", dpi=300, bbox_inches="tight")
# %%
record = db.session.get(spidb.Record, 2236)
# audios = detection.retreive_acoustic_data(db, record.sensor, start=record.start, end=record.end, channels=[0, 1, 2, 3, 7], duration=60)

fig, ax = visualizer.spectra_display(db, start=record.start, end=record.end, sensor=record.sensor, section=[0, 1, 2, 3, 7], spl=True)
ax.set_xlim(0, 8000)
ax.set_ylim(-125, -25)
ax.set_yticks([-125, -75, -25])

# audio = db.get_audio(record.start, record.end, sensor=record.sensor, channel_number=7)


# update the label of the last line in the plot

fig.savefig(r"projects/Dissertation/proposal/figures/external_noise/cfb_spectra_display.pdf", dpi=300, bbox_inches="tight")
# %%
