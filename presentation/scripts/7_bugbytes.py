# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cblind

from scipy import stats, signal
from dankpy import color
from sonicdb import audio 

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

pd.options.mode.chained_assignment = None

plt.style.use("dankpy.styles.stevens_presentation")
#%%
targets = [
    {
        'target': "Sitophilus oryzae",
        'material': "Wheat Groats",
        "file": r"data/bug_bytes/A7-piezorw1.wav",
        'sensor': "piezoelectric", 
        "low-frequency": 500, 
        "high-frequency": 8000,
        "low-amp": -110,
        "high-amp": -50,
        "top-freq": 10000
    },
]
targets = pd.DataFrame(targets)

for i, t in targets.iterrows():

    a = audio.Audio(t["file"])

    fig, ax = a.plot_spectrogram(
        window="hann",
        window_size=1024,
        nfft=1024,
        nperseg=1024,
        noverlap=512,
        time_format="seconds",
        zmin=t["low-amp"],
        zmax=t["high-amp"],
        showscale="top",
        cmap=matplotlib.colormaps.get_cmap("cb.solstice"),
    )
    ax.set_ylim(0, t["top-freq"])
    ax.set_yticks([0, round(t["top-freq"]/2), t["top-freq"]])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.set_size_inches(3.61, 4.21)

    f, p = signal.welch(
        a.data.signal,
        fs=a.sample_rate,
        nperseg=1024,
        window="blackmanharris",
        scaling="spectrum",
        average="mean"
    )

    a.bandpass_filter(t["low-frequency"], t["high-frequency"], order=10, overwrite=True)
    a.envelope(overwrite=True)

    level = 1*np.median(a.data.signal)
    noise = a.data[a.data.signal <= level]
    rms = np.sqrt(np.mean(noise.signal**2))

    f_n, p_n = signal.welch(
        noise.signal,
        fs=a.sample_rate,
        nperseg=1024,
        window="blackmanharris",
        scaling="spectrum",
        average="mean"
    )

    fig, ax = plt.subplots()
    ax.plot(f, 10*np.log10(p), label="Signal", color=color.colors[0], linestyle="solid")
    ax.plot(f_n, 10*np.log10(p_n), label="Reference")
    ax.set_xlim(500, 8000)
    ax.set_xticks([500, 4250, 8000])
    ax.set_ylim(-100, 0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectral Power [dB]")
    ax.legend(loc="upper right")
    fig.set_size_inches(3.61, 4.21)
    # fig.savefig(rf"projects/Dissertation/proposal/figures/bug_bytes/{t['sensor']} - {t['target']}_spectra.pdf", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(f, 10*np.log10(p) - 10*np.log10(p_n))
    ax.set_xlim(0, round(max(f)))
    ax.set_ylim(None, None)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("SNR [dB]")
    fig.set_size_inches(3.61, 4.21)

    # fig.savefig(rf"projects/Dissertation/proposal/figures/bug_bytes/{t['sensor']} - {t['target']}_snr.pdf", dpi=300)


    p = p[(f>=t["low-frequency"]) & (f<=t["high-frequency"])]
    p = 10*np.log10(p)
    spl = 10*np.log10(np.sum(10**(p/10)))

    p_n = 10*np.log10(p_n)
    p_n = p_n[(f_n>=t["low-frequency"]) & (f_n<=t["high-frequency"])]
    npl = 10*np.log10(np.sum(10**(p_n/10)))


    snr = spl-npl
    print(f"{t['target']} is {snr} dB")

    targets.at[i, "NSEL"] = snr

    a.data.signal = a.data.signal / rms

    min_threshold = 0.5*np.max(a.data.signal)
    max_threshold = 0.9*np.max(a.data.signal)
    rms = np.sqrt(np.mean(a.data.signal[(a.data.signal >= min_threshold) & (a.data.signal <= max_threshold)]**2))

    targets.at[i, "NSPA"] = 20*np.log10(rms)

    fig, ax = plt.subplots()
    ax.plot(a.data.seconds, a.data.signal)
    ax.hlines([rms], 0, round(a.data.seconds.max()), color="red", linestyle="dashed", label="NSPA ({:.0f} dB)".format(20*np.log10(rms)))
    # plot rectange at cutoffs 
    ax.fill_between(a.data.seconds, min_threshold, max_threshold, color="red", alpha=0.25, label="Analysis Region")
    ax.legend(loc="upper right")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_ylim(0, 200)
    ax.set_yticks([0, 100, 200])
    ax.set_xlim(0, round(a.data.seconds.max()))
    # fig.savefig(rf"projects/Dissertation/proposal/figures/bug_bytes/{t['sensor']} - {t['target']}_envelope.pdf", dpi=300)
    # fig.savefig(rf"projects/Dissertation/proposal/figures/bug_bytes/{t['sensor']} - {t['target']}_envelope.jpeg", dpi=300)
    fig.set_size_inches(3.61, 4.21)

    # document.add_figure(doc, fig, ax, "Envelope", add_break=False)
    # doc.add_page_break()
#%%
# targets.NSPA = targets.NSPA.round(2)
# targets.NSEL = targets.NSEL.round(2)

# targets.to_csv(r"projects/Dissertation/proposal/figures/bug_bytes/2024_06_28_bug_bytes.csv", index=False)
# targets = targets.drop(columns=["file"])
# document.add_dataframe(doc, targets)
# doc.save(r"projects/Dissertation/proposal/figures/bug_bytes/2024_06_28_bug_bytes.docx")
#%%
# targets = pd.read_csv(r"projects/Dissertation/proposal/figures/bug_bytes/2024_06_28_bug_bytes.csv")