# %%
from archive import lookup
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal

from dankpy import color, dt

from spidb import spidb, visualizer, detection
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
# %%
targets = [
    {
        "target": "Darkling beetle",
        "material": "Rice",
        "test": 83,
        "iteration": 2,
        "channel": 0,
        "record": 950,
    },
    {
        "target": "Mealworm",
        "material": "Wheat Groats",
        "test": 93,
        "iteration": 12,
        "channel": 2,
        "record": 1007
    },
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "test": 163,
        "iteration": 6,
        "channel": 0,
        "record": 2236
    },
    {
        "target": "Bean beetle",
        "material": "Oatmeal",
        "test": 7,
        "iteration": 1,
        "channel": 1,
        "record": 40
    },
    {
        "target": "Noise",
        "material": "60 dBA",
        "channel": 0,
        "record": 460,
        "amplitude": 200,
    },
    {
        "target": "Noise",
        "material": "90 dBA",
        "channel": 0,
        "record": 463,
        "amplitude": 1000,
    },
]

# %%
for target in targets[-1:]:
    record = db.session.get(spidb.Record, target["record"])
    fig, ax = visualizer.spectrogram_display(
        db,
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        time_format="seconds",
        section="internal",
        size=(11.71, 4.24),
    )
    for x in ax:
        x.set_ylim(0, 12000)
        x.set_yticks([0, 6000, 12000])
        # update the legend so its center right



    # fig.suptitle("$\mathit{Tenebrio\ molitor}$ in rice")
    # set the cbar ticks

    # fig.savefig(
        # "projects/Dissertation/presentation/figures/acoustic_signatures/{}_{}_spectrogram_display.jpeg".format(
            # target["target"], target["material"]
        # ),
        # dpi=300,
    # )

# %%
for target in targets:
    record = db.session.get(spidb.Record, target["record"])
    fig, ax = visualizer.spectrogram_display(
        db,
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        time_format="seconds",
        section="minimal",
        size=(6, 3),
    )
    for x in ax:
        x.set_ylim(0, 12000)
        x.set_yticks([0, 6000, 12000])
    # set the cbar ticks
    ax.suptitle("$\mathit{Tenebrio\ molitor}$ in rice")
    fig.savefig(
        "projects/Dissertation/presentation/figures/acoustic_signatures/complete_{}_{}_spectrogram_display.jpeg".format(
            target["target"], target["material"]
        ),
        dpi=300,
    )

    break

# %%
audios = db.get_audios(
    start=record.start, end=record.end, sensor=record.sensor, channels=[0, 1, 2, 3]
)

fig, ax = visualizer.spectra_display(audios)
for i, line in enumerate(ax.lines):
    line.set_color(colors[i])
    line.set_label(f"Ch. {i}")
ax.plot(
    noise["frequency"],
    noise["average_db"],
    color="black",
    linestyle="solid",
    label="Reference",
    zorder=10,
)
ax.set_xlim(0, 12000)
# make the lines different colors and make the labels match
ax.legend(loc="upper right", ncols=5)
# make the ylabel smaller
ax.set_ylabel("Spectral Power [dB]")
ax.set_ylim(-125, -25)
ax.set_yticks([-125, -100, -75, -50, -25])
ax.set_title("$\mathit{Tenebrio\ molitor}$ in rice")
fig.savefig(
    "projects/Dissertation/presentation/figures/acoustic_signatures/{}_{}_spectra_display.jpeg".format(
        targets[0]["target"], targets[0]["material"]
    ),
    dpi=300,
)
# %%


# %%
for i, t in enumerate([targets[0], targets[2], targets[3]]):
    record = db.session.get(spidb.Record, t["record"])
    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    fig, ax = audio.plot_spectrogram(
        window_size=1024,
        nperseg=1024,
        nfft=1024,
        noverlap=512,
        zmin=-140,
        zmax=-80,
        time_format="seconds",
        showscale="right",
    )

    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 4000, 8000])
    # add label to legend
    ax.plot([], [], " ", label=f"Piezoelectric - Ch. {t['channel']}")
    ax.legend(handlelength=0, handletextpad=0)

    fig.savefig(
        "projects/Dissertation/presentation/figures/acoustic_signatures/{}_{}_spectrogram.jpeg".format(
            t["target"], t["material"]
        ),
        dpi=300,
    )
# %%
fig, ax = plt.subplots()
for i, t in enumerate([targets[0], targets[2], targets[3]]):
    record = db.session.get(spidb.Record, t["record"])

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    f, p = signal.welch(
        audio.data.signal,
        fs=audio.sample_rate,
        nperseg=1024,
        noverlap=512,
        window="blackmanharris",
        scaling="spectrum",
    )
    p = 10 * np.log10(p)

    ax.plot(
        f,
        p,
        label=f"{lookup.lookup(key=t['target'], latex=True, min=True)} ({t['material']})",
        color=colors[i],
    )
ax.plot(
    noise["frequency"],
    noise["average_db"],
    color="black",
    linestyle="solid",
    label="Reference",
    zorder=10,
)
ax.set_xlim(0, 8000)
ax.set_ylim(-125, -25)
ax.set_yticks([-125, -75, -25])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Spectral Power [dB]")
ax.legend(
    bbox_to_anchor=(0.0, 1.10, 1, 0),
    loc="lower center",
    ncol=4,
    columnspacing=1,
    markerscale=0.5,
)
fig.savefig(
    "projects/Dissertation/presentation/figures/acoustic_signatures/spectra.jpeg",
    dpi=300,
)
# %%
fig, ax = plt.subplots()
for i, t in enumerate(targets[:4]):
    record = db.session.get(spidb.Record, t["record"])

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    f, p = signal.welch(
        audio.data.signal,
        fs=audio.sample_rate,
        nperseg=1024,
        noverlap=512,
        window="blackmanharris",
        scaling="spectrum",
    )
    p = 10 * np.log10(p)
    snr = p - 10 * np.log10(noise[f"{t['channel']}"])

    ax.plot(
        f,
        snr,
        label=f"{lookup.lookup(key=t['target'], latex=True, min=True)} ({t['material']})",
        color=colors[i],
    )
ax.set_xlim(0, 8000)
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("SNR [dB]")
ax.legend(
    bbox_to_anchor=(0.0, 1.10, 1, 0),
    loc="lower center",
    ncol=2,
    columnspacing=1,
    markerscale=0.5,
)
fig.savefig(
    "projects/Dissertation/presentation/figures/acoustic_signatures/snr.jpeg", dpi=300
)
# %%
for i, t in enumerate(targets[-1:]):
    fig, ax = plt.subplots(figsize=(3.61, 4.21))
    record = db.session.get(spidb.Record, t["record"])

    audio = db.get_audio(
        start=record.start,
        end=record.end,
        sensor=record.sensor,
        channel_number=t["channel"],
    )

    audio.fade_in(1, overwrite=True)
    audio.fade_out(1, overwrite=True)

    # audio.bandpass_filter(500, 6000, 10, overwrite=True)

    ax.set_title("Raw Signal")
    ax.plot(
        audio.data.seconds,
        audio.data.signal,
        label=f"Piezoelectric - Ch. {t['channel']}",
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    # ax.set_yticks([-0.5, 0, 0.5])
    # ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)
    fig.savefig(
        "projects/Dissertation/presentation/figures/acoustic_signatures/{}_{}_waveform_filtered.jpeg".format(
            t["target"], t["material"]
        ),
        dpi=300,
    )
    break
#%%
    audio.envelope(overwrite=True)
    fig, ax = plt.subplots(figsize=(3.61, 4.21))
    ax.plot(
        audio.data.seconds,
        audio.data.signal,
        label=f"Piezoelectric - Ch. {t['channel']}",
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    # ax.set_title("Filtered and Enveloped Signal")
    ax.legend(loc="upper right", handlelength=0, handletextpad=0)
    fig.savefig(
        "projects/Dissertation/presentation/figures/acoustic_signatures/{}_{}_waveform_envelope.jpeg".format(
            t["target"], t["material"]
        ),
        dpi=300,
    )

# %%
