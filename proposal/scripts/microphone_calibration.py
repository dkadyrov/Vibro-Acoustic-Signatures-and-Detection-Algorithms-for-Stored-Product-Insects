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
#%%
events = [35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49] 


events = pd.DataFrame([db.session.get(spidb.Event, event).__dict__ for event in events])

sensor = db.session.get(spidb.Sensor, 1)

# get the number out of event["description"]
events["measured"] = events["description"].str.extract(r"(\d+\.?\d*)").astype(float)
# %%
spls = []
spls2 = [] 

for e, group in events.groupby("measured"):
    for i, row in group.iterrows():
        audio = db.get_audio(row["start"], row["end"], sensor=sensor, channel_number=7)

        # fig, ax = audio.plot_spectrogram(window_size=1024, nperseg=1024, nfft=1024, noverlap=512, time_format="seconds", zmin=-125, zmax=-80)
        # ax.set_ylim(0, 8000)
        # ax.set_title(f"{row['description']}")

        fs = audio.sample_rate

        f_b = 20.598997
        f_c = 12194.217
        num1, den1 = bilinear([1, 0, 0], [1, 2 * np.pi * (f_b + f_c), (2 * np.pi)**2 * f_b * f_c], fs)

        # Section 2: High-frequency filter
        f_a = 107.65265
        f_d = 737.86223
        num2, den2 = bilinear([1, 0, (2 * np.pi * f_a)**2], [1, 2 * np.pi * (f_a + f_d), (2 * np.pi)**2 * f_a * f_d], fs)
        
        # A-weighting filter gain at 1000 Hz should be 0 dB. The ideal transfer function
        # has a gain of approximately -2.0 dB at 1000 Hz, so we apply a correction factor.
        gain_correction = 10**(2.0/20.0)

        # Apply the two filter sections in series.
        weighted_signal = lfilter(num1, den1, audio.data.signal)
        weighted_signal = lfilter(num2, den2, weighted_signal)
        
        # Apply the gain correction to achieve 0 dB at 1000 Hz.
        weighted_signal *= gain_correction

        # Calculate the RMS value of the A-weighted signal.
        rms = np.sqrt(np.mean(weighted_signal**2))
        
        # Define the reference pressure for SPL calculation.
        p_ref = 20e-6

        # Calculate SPL in dB.
        spl = 20 * np.log10(rms / p_ref)




        # spl = acoustics.calculate_spl_dba(audio.data.signal, audio.sample_rate)

        # spl2 = acoustics.a_weighting_spl(audio.data.signal, audio.sample_rate)

        spls.append(spl)
        # spls2.append(spl2)

events["spl"] = spls
# events["spl2"] = spls2
#%%
fig, ax = plt.subplots()
ax.scatter(events["measured"], events["spl"], label="SPL")
ax.scatter(events["measured"], events["spl2"], label="SPL2")
# %%
