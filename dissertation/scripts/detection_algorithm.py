# %%
from datetime import timedelta

from . import lookup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dankpy import acoustics, color, document
from scipy import signal

from spidb import detection, normalization, spidb, visualizer

plt.style.use("dankpy.styles.mdpi")

db = spidb.Database(r"data/spi.db")

targets = [
    {
        "target": "Darkling beetle",
        "material": "Rice",
        "channel": 0,
        "record": 950,
        "amplitude": 2000,
    },
    {
        "target": "Mealworm",
        "material": "Wheat Groats",
        "channel": 2,
        "record": 1007,
        "amplitude": 500,
    },
    {
        "target": "Confused flour beetle",
        "material": "Flour",
        "channel": 0,
        "record": 2236,
        "amplitude": 200,
    },
    {
        "target": "Bean beetle",
        "material": "Oatmeal",
        "channel": 1,
        "record": 40,
        "amplitude": 200,
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
        "amplitude": 500,
    },
]
filters = [[1565, 6000]]
for filt in filters:
    for i, target in enumerate(targets[-1:]):
        record = db.session.get(spidb.Record, target["record"])
        # fig, ax = visualizer.waveform_display(
        #     db=db,
        #     start=record.start,
        #     end=record.end,
        #     sensor=record.sensor,
        #     time_format="seconds",
        #     normalize="noise",
        #     envelope=True,
        #     filter=filt,
        #     external_spl=True,
        # )
        # for a in ax:
        #     a.set_ylim(0, target["amplitude"])

        # fig.savefig(f"projects/MDPI-Detection/report/figures/waveforms/{target['target']}-{target['material']}-_{filt[0]}-{filt[1]}.pdf", bbox_inches="tight", dpi=300)

        audio = db.get_audio(record.start, record.end, sensor=record.sensor, channel_number=target["channel"])

    

        audio.bandpass_filter(filt[0], filt[1], order=10, overwrite=True)
        audio.envelope(overwrite=True)
        coef =  normalization.noise_coefficient(db, record.sensor, record.sensor.channels[target["channel"]], "bandpass", filt[0], filt[1], order=10)
        
        audio.data.signal = audio.data.signal / coef

        fig, ax = plt.subplots()
        ax.plot(audio.data.seconds, audio.data.signal)
        ax.plot([], [], label=f"Ch. {target['channel']}")

        ax.set_xlim(0, 60)
        ax.set_ylim(0, None)
        ax.set_ylabel("Normalized\nAmplitude")
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right", handlelength=0, handletextpad=0)

        # fig.savefig(f"projects/MDPI-Detection/report/figures/waveforms/{target['target']}-{target['material']}-_{filt[0]}-{filt[1]}_Ch{target['channel']}.pdf", bbox_inches="tight", dpi=300)
        break
        plt.close(fig)
#%%