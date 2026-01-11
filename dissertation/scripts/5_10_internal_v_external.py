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

plt.style.use("dankpy.styles.latex")
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

noise_internal = processing.process_signal(
    db, noise_internal, channel=noise_record.sensor.channels[0]
)

T = 21.6  # dB
K = 30.54
A = 10 ** (T / 20) * 10 ** (6 / 20)
KT = 10 ** (T / 20) * 10 ** (K / 20) / 10 ** (6 / 20)

nspa_internal = np.sqrt(
    np.mean(noise_internal.data.signal[noise_internal.data.signal >= A] ** 2)
)
nspa_internal = 20 * np.log10(nspa_internal)

nspa_external = np.sqrt(
    np.mean(noise_external.data.signal[noise_external.data.signal >= KT] ** 2)
)
nspa_external = 20 * np.log10(nspa_external)

fig, ax = plt.subplots(nrows=2, figsize=(6, 2.5), sharex=True)
ax[0].plot(
    noise_internal.data.seconds,
    noise_internal.data.signal,
    # label=f"Ch. 0, NSPA {noise_internal_nspa:.2f} dB",
)
ax[0].add_patch(
    plt.Rectangle(
        (0, A),
        150,
        150 - A,
        facecolor=color.colors[3],
        alpha=0.5,
        label="$A_i$ Threshold",
    )
)
ax[0].legend(
    loc="upper right",
    # handlelength=0,
    # handletextpad=0,
    title=f"Ch. 0, NSPA {nspa_internal:.2f} dB",
    # title_fontsize=8,
)
ax[0].set_ylim(0, 150)
ax[0].set_yticks([0, 75, 150])

ax[1].plot(
    noise_external.data.seconds,
    noise_external.data.signal,
    # label=f"Ch. 7, NSPA {nspa_noise_external:.2f} dB",
)
# ax[1].axhline(y=K * 30, color="r", linestyle="--", label=f"Threshold = {K:.2f}")
ax[1].add_patch(
    plt.Rectangle(
        (0, KT),
        60,
        2500 - KT,
        facecolor=color.colors[0],
        alpha=0.5,
        label="$A_e$ Threshold",
    )
)

ax[1].legend(
    loc="upper right",
    title=f"Ch. 7, NSPA {nspa_external:.2f} dB",
    # title_fontsize=8,
)
ax[1].set_ylim(0, 2500)
ax[1].set_yticks([0, 1250, 2500])
ax[1].set_xlim(0, 20)
fig.supylabel("Normalized Amplitude")
fig.supxlabel("Time [s]")
# mark the threshold line
# ax.axhline(y=amp_threshold, color="r", linestyle="--")
# mark where the signal is removed

fig.savefig(
    r"projects\Dissertation\dissertation\figures\5_10_noise_internal_vs_external.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
