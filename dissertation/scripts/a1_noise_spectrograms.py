# %%
import cblind as cb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aspids_tools import visualizer
from dankpy import color, dt
from matplotlib import pyplot as plt
from scipy import signal

from . import lookup
from spidb import spidb
from sonicdb import audio

pd.options.mode.chained_assignment = None

plt.style.use("dankpy.styles.latex")

audios = [
    {
        "target": "Crowd",
        "file": r"data\external\Noise\crowd.flac"
    },
    {
        "target": "Factory 1",
        "file": r"data\external\Noise\factory1.wav"
    },
    {
        "target": "Factory 2",
        "file": r"data\external\Noise\factory2.wav"
    },
    {
        "target": "Male Speech",
        "file": r"data\external\Noise\malespeech.mp3"
    },
]

for i, a in enumerate(audios):
    a = audio.Audio(a["file"])
    a = a.trim(start=0, length=60, time_format="seconds")
    fig, ax = a.plot_spectrogram(time_format="seconds", showscale=False, zmin=-100, zmax=-50, cmap=cb.cbmap("cb.solstice"))
    ax.set_ylim(0, 10000)
    ax.set_yticks([0, 5000, 10000])
    fig.set_size_inches(6, 2.5)
    fig.savefig(rf"projects\Dissertation\dissertation\figures/a{i}_spectrogram.pdf", dpi=300)
#%%

# alpha = ["a", "b"]

# for i, t in enumerate(targets):
#     record = db.session.get(spidb.Record, t["record"])
#     audio = db.get_audio(
#         start=record.start,
#         end=record.end,
#         sensor=record.sensor,
#         channel_number=t["channel"],
#     )

#     fig, ax = audio.plot_spectrogram(
#         window_size=1024,
#         nperseg=1024,
#         nfft=1024,
#         noverlap=512,
#         zmin=-140,
#         zmax=-80,
#         time_format="seconds",
#         showscale="right",
#         cmap=cb.cbmap("cb.solstice"),
#     )

#     ax.set_ylim(0, 8000)
#     ax.set_yticks([0, 4000, 8000])
#     # add label to legend
#     ax.plot([], [], " ", label=f"Ch. {t['channel']}")
#     ax.legend(handlelength=0, handletextpad=0)

#     fig.savefig(
#         r"projects\Dissertation\dissertation\figures/3{}_spectrogram.pdf".format(
#             alpha[i]
#         ),
#         dpi=300,
#     )
# # %%
