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

records = [463]

for record in records: 
    record = db.session.get(spidb.Record, record)
    audios = db.get_audios(record.sensor, start=record.start, end=record.end, channels=[4,5,7])

    fig, ax = visualizer.spectra_display(audios, spl=True)

    reference = normalization.reference_signal(db, record.sensor, channels=[4,5])

    ax.plot(reference["frequency"], reference["average"], label="Reference", color="black", linestyle="-")

    ax.legend(loc="upper right", ncols=4)
    ax.set_ylim(-125, -25)
    ax.set_yticks([-125, -75, -25])
    ax.set_xlim(0, 8000)
    
    fig.savefig(f"projects/Dissertation/proposal/figures/insulation_suppression/spectra_{record.external_spl:.2f}.pdf", dpi=300)
#%%
results = {}
for audio in audios:
    audio.fade_in(fade_time=1, overwrite=True)
    audio.fade_out(fade_time=1, overwrite=True)

    f, p = signal.welch(
        audio.data.signal,
        fs=audio.sample_rate,
        nperseg=1024,
        nfft=1024,
        noverlap=512,
        window="blackmanharris",
        average="mean",
        scaling="spectrum",
    )

    p = 10 * np.log10(p)

    results[f"Ch. {audio.channel_number}"] = p

results = pd.DataFrame(results)
results["frequency"] = f
results["Average"] = (results["Ch. 4"] + results["Ch. 5"]) / 2
results = results[results["frequency"] >= 1800]
results = results[results["frequency"] <= 8000]
difference = (results["Ch. 7"] - results["Average"])

print(f"The maximum difference is {difference.max()}")
print(f"The average difference is {difference.mean()}")
# %%
reference = normalization.reference_signal(db, record.sensor, channels=[4,5,7])

fig, ax = plt.subplots()
for i, audio in enumerate(audios):
    audio.fade_in(fade_time=1, overwrite=True)
    audio.fade_out(fade_time=1, overwrite=True)

    f, p = signal.welch(
        audio.data.signal,
        fs=audio.sample_rate,
        nperseg=1024,
        nfft=1024,
        noverlap=512,
        window="blackmanharris",
        average="mean",
        scaling="spectrum",
    )

    p = 10 * np.log10(p)
    snr = p - reference[f"Ch. {audio.channel_number}"]

    ax.plot(f, snr, label=f"Ch. {audio.channel_number}", color=color.colors[i])

ax.legend(loc="upper right", ncols=4)
ax.set_xlim(0, 12000)
ax.set_ylim(0, 70)
ax.set_yticks([0, 35, 70])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("SNR [dB]")

fig.savefig(f"projects/Dissertation/proposal/figures/insulation_suppression/snr_{record.external_spl:.2f}.pdf", dpi=300)
# %%
