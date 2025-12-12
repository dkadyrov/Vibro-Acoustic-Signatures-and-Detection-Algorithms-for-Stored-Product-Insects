# %%
from spidb import spidb, normalization, visualizer, detection
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dankpy import acoustics, color, document
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal

plt.style.use("dankpy.styles.mdpi")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")
#%%

# docx = document.Document("projects/Dissertation/proposal/microphone_calibration.docx")

targets = [
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "name": "$\mathit{T. confusum}$ (Flour)",
        "channel": 0,
        "record": 2236
    },
    {
        "target": "Bean beetle",
        "material": "Oatmeal",
        "name": "$\mathit{C. maculatus}$ (Oatmeal)",
        "channel": 1,
        "record": 40
    }
]

for target in targets:
    record = db.session.get(spidb.Record, target["record"])
    audio = db.get_audio(start=record.start, end=record.end, sensor=record.sensor, channel_number=target["channel"])

    f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")

    p = 10 * np.log10(p)
    target["frequency"] = f
    target["spectrum"] = p

insect_data = pd.DataFrame(targets)


# events = [580, 438, 534, 435, 544, 582, 579, 542, 523, 442, 585, 547, 545, 522, 550, 546, 548, 464, 549, 554]
events = [
435,
438,
439,
443,
456,
458,
459,
471,
479,
492,
579,
580,
581,
582,
583 
    ]
#%% Plot Spectra Comparison
for event in events[11:]: 
    record = db.session.get(spidb.Record, event)
    audios = db.get_audios(start=record.start, end=record.end, sensor=record.sensor, channels=[0, 1, 2, 3, 7])

    nspas = [record.classifications[i].classification for i in range(4)]
    # find which index is the maximum
    max_nspa_index = nspas.index(max(nspas))
    print(f"Max NSPA Channel: {max_nspa_index} - {record.classifications[max_nspa_index].classification} dB")

    audio = audios[max_nspa_index]

    fig, ax = plt.subplots()
    # for i, target in enumerate(targets):
    #     record = db.session.get(spidb.Record, target["record"])
    #     audio = db.get_audio(start=record.start, end=record.end, sensor=record.sensor, channel_number=target["channel"])

    f, p = signal.welch(audio.data.signal, fs=audio.sample_rate, nperseg=1024, noverlap=512, window="blackmanharris", scaling="spectrum")

    p = 10 * np.log10(p)

    ax.plot(f, p, label=f"Noise (Ch. {max_nspa_index}, NSPA {float(record.classifications[max_nspa_index].classification):.2f} dB)", color="black", linestyle="-")

    for i, target in insect_data.iterrows():
        ax.plot(target["frequency"], target["spectrum"], label=target["name"], color=color.colors[i])

    ax.plot(noise["frequency"], 10 * np.log10(noise["0"]), label="Reference", color="black", linestyle="--")

    ax.set_xlim(0, 8000)
    ax.set_ylim(-125, -25)
    ax.set_yticks([-125, -75, -25])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectral Power [dB]", fontsize=8)
    # put the legend on top
    ax.legend(loc="upper right", fontsize=8, ncols=2, framealpha=0.8)
    fig.set_size_inches(6, 1.35)


    fig, ax = visualizer.waveform_display(db, start=record.start, end=record.end, sensor=record.sensor, time_format="seconds", external_spl=False, envelope=True, normalize="noise", filter=[1565, 6000])
    fig.set_size_inches(6, 3)

    fig.show() 

    fig, ax = visualizer.spectrogram_display(db, start=record.start, end=record.end, sensor=record.sensor, time_format="seconds", section="minimal", showscale="right", zmin=-140, zmax=-80)
    for a in ax:
        a.set_ylim(0, 8000)
        a.set_yticks([0, 4000, 8000])
    # update figure size
    fig.set_size_inches(6, 3)

    fig.show()

    for a in ax:
        a.set_ylim(0, 8000)
    
    data, acoustic_data = detection.acoustic_detection(audios, IT=19, ET=19, DR=20, NR=20, internal_filt=[1565, 6000], external_filt=500, scaling_i=6, scaling_e=4, return_audio=True, db=db)
    fig, ax = detection.acoustic_detection_display(data, acoustic_data, time_format="seconds", spl=False)

    fig.set_size_inches(6, 3.4)
    fig.show()

    break
#%%

# for event in events: 


# event = db.session.get(spidb.Record, 377)

# start = event.start
# end = start + timedelta(seconds=60)

# fig, ax = visualizer.waveform_display(db, start=start, end=end, sensor=event.sensor, time_format="seconds", external_spl=True, envelope=True, normalize="noise", filter=[1565, 6000], size=(5.67,4.21))
# for i, a in enumerate(ax):
#     # if i < len(ax)-1:
#     a.set_ylim(0, 50)
#     # a.set_yticks([0, 50])
#     # else:
#         # a.set_ylim(0, 500)
#         # a.set_yticks([0, 500])

#     # a.set_yticks([0, 2500, 5000])
#     # update legend in a to have smaller font 
#     a.legend(fontsize=10, loc="upper right", handlelength=0, handletextpad=0)
# # fig.savefig(f"projects/Dissertation/proposal/figures/external_noise/waveform_{event.id}.pdf", dpi=300)
# #%%
# audios = db.get_audios(event.sensor, start=start, end=end, channels=[0, 1, 2, 3, 6])

# fig, ax = visualizer.spectra_display(audios)

# #%%
# %%
