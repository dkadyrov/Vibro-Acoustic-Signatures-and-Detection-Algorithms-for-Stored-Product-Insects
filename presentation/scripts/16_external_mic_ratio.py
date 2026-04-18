# %%


import cblind as cb  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
from spidb import spidb

from aspids_tools import normalization

plt.style.use("dankpy.styles.stevens_presentation")
# noise = pd.read_pickle(r"data/noise_500-6000.pkl")

db = spidb.Database(r"data/spi.db")


records = [460, 461, 462, 463]

# %%
maxes = [[10, 100], [20, 200], [30, 500], [60, 1000]]
titles = [60, 70, 80, 90]
fig, axf = plt.subplots(ncols=2, nrows=4, figsize=(5.67, 4.21))

amplitudes = []


for r, record in enumerate(records):
    axs = axf[r]

    record = db.session.get(spidb.Record, record)

    channels = [3, 7]

    labels = [
        "Ch. 3 (Piezoelectric)",
        "Ch. 7 (External Microphone)",
    ]

    audios = db.get_audios(
        record.sensor,
        start=record.start,  # + timedelta(seconds=2.75),
        end=record.end,  # + timedelta(seconds=3.25),
        channels=channels,
    )

    for i, audio in enumerate(audios):
        ax = axs[i]

        channel = record.sensor.channels[channels[i]]

        nspa = normalization.calculate_nspa(
            audio,
            filter="bandpass",
            low=1565,
            high=6000,
            normalize="noise",
            channel=channel,
            db=db,
        )

        audio.fade_in(1, overwrite=True)
        audio.fade_out(1, overwrite=True)
        audio.bandpass_filter(1565, 6000, order=10, overwrite=True)

        audio.envelope(overwrite=True)

        audio = normalization.noise_normalize(
            db,
            audio,
            channel=record.sensor.channels[channels[i]],
            filter="bandpass",
            low=1565,
            high=6000,
            coefficient="set",
        )
        max_amplitude = audio.data.signal.max()

        ax.plot(audio.data.seconds, audio.data.signal, label=f"NSPA {nspa:.0f} dB")
        if r == 0:
            ax.set_title(f"Channel {channels[i]}", fontsize=10)
        # ax.set_xlabel("Time [s]")
        ax.set_xlim(0, 60)
        if i == 0:
            ax.set_ylabel(f"{titles[r]} dBA", fontsize=10)


            # ax.set_ylabel("Normalized\nAmplitude")
            # ax.set_ylim(0, None)
            ax.set_ylim(0, maxes[r][0])
            ax.set_yticks([0, round(maxes[r][0] / 2), maxes[r][0]])
        else:
            ax.set_ylim(0, maxes[r][1])
            ax.set_yticks([0, round(maxes[r][1] / 2), maxes[r][1]])
            # ax.set_ylim(0, 1000)
            # ax.set_yticks([0, 500, 1000])
        ax.legend(loc="upper right", handlelength=0, handletextpad=0, fontsize=8)
        # if f == 0:
        #     ax.set_ylim(0, t["amplitude"])
        #     ax.set_yticks([0, round(t["amplitude"] / 2), t["amplitude"]])
        # else:
        #     ax.set_ylim(0, t["amplitude2"])
        #     ax.set_yticks([0, round(t["amplitude2"] / 2), t["amplitude2"]])
fig.supxlabel("Time [s]")
fig.supylabel("Normalized Amplitude")
# %%
channels = [0, 1, 2, 3, 7]
data = []
for r, record in enumerate(records):
    record = db.session.get(spidb.Record, record)
    nspas = [record.classifications[c] for c in channels]
    nspas = [float(nspa.classification) for nspa in nspas]

    data.append(
        {
            "SPL": round(record.external_spl, 2),
            "Ch. 0 ": nspas[0],
            "Ch. 1 ": nspas[1],
            "Ch. 2 ": nspas[2],
            "Ch. 3 ": nspas[3],
            "Ch. 7 ": nspas[4],
            "K0": nspas[4] - nspas[0],
            "K1": nspas[4] - nspas[1],
            "K2": nspas[4] - nspas[2],
            "K3": nspas[4] - nspas[3],
        }
    )
data = pd.DataFrame(data)

# %%
