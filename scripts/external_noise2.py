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

    start = event.start
    end = start + timedelta(seconds=60)

    fig, ax = plt.subplots()

    nspas = []
    nsels = []
    for i, channel in enumerate(event.sensor.channels[:4]):
        audio = db.get_audio(start, end, sensor=event.sensor, channel_number=channel.number)

        f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")

        p = 10 * np.log10(p)
        snr = p - 10*np.log10(noise[f"{channel.number}"])

        ax.plot(f, snr, label=f"Ch. {channel.number}", color=color.colors[i])

        nspa = normalization.calculate_nspa(audio, filter="bandpass", low=500, high=6000, channel=channel)
        nspas.append(nspa)

        nsel = normalization.calculate_nsel(audio, filter="bandpass", low=500, high=6000, channel=channel, db=db)
        nsels.append(nsel)

    audio = db.get_audio(start, end, sensor=event.sensor, channel_number=7)

    f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")

    p = 10 * np.log10(p)
    snr = p - 10*np.log10(noise["7"])

    spl = acoustics.calculate_spl_dba(audio.data.signal, audio.sample_rate)
    spl = normalization.spl_coefficient(spl)

    ax.plot(f, snr, label=f"Ch. 7 (SPL {spl:.2f} dBA)", color="black")   

    ax.set_xlim(0, 8000)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 35, 70])
    ax.set_ylabel("SNR [dB]")
    ax.legend(loc="upper right", ncols=5)

    nspa = normalization.calculate_nspa(audio, filter="bandpass", low=500, high=6000, channel=event.sensor.channels[-1])

    nsel = normalization.calculate_nsel(audio, filter="bandpass", low=500, high=6000, channel=event.sensor.channels[-1], db=db)

    fig.savefig(rf"projects/Dissertation/proposal/figures/external_noise/snr_internal_{spl:.0f}.pdf", dpi=300)

    d = {
        "spl": spl,
        "NSPA (Internal)": max(nspas),
        "NSEL (Internal)": max(nsels),
        "NSPA (Ch. 7)": nspa,
        "NSEL (Ch. 7)": nsel
    }

    data.append(d)

data = pd.DataFrame(data)
#%%