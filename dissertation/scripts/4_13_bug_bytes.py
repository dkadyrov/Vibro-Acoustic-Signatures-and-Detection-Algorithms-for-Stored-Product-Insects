# %%
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from scipy import signal
from dankpy import color, audio  # type: ignore

import cblind

plt.style.use("dankpy.styles.latex")
# %%
targets = [
    {
        "target": "Plodia interpunctella",
        "material": "Dog Food",
        "file": r"data/bug_bytes/A1-dfaccs11.wav",
        "sensor": "accelerometer",
        "low-frequency": 2000,
        "high-frequency": 3000,
        "low-amp": -120,
        "high-amp": -60,
        "top-freq": 5000,
    },
    {
        "target": "Plodia interpunctella",
        "material": "Dog Food",
        "file": r"data/bug_bytes/A2-piezimm1.wav",
        "sensor": "piezoelectric",
        "low-frequency": 400,
        "high-frequency": 1200,
        "low-amp": -90,
        "high-amp": -30,
        "top-freq": 3000,
    },
    {
        "target": "Sitophilus oryzae",
        "material": "Wheat Groats",
        "file": r"data/bug_bytes/A3-pvdfrw1.wav",
        "sensor": "PVDF",
        "low-frequency": 4000,
        "high-frequency": 11000,
        "low-amp": -130,
        "high-amp": -70,
        "top-freq": 12500,
    },
    {
        "target": "Sitophilus oryzae",
        "material": "Wheat Groats",
        "file": r"data/bug_bytes/A4-accelrw.wav",
        "sensor": "accelerometer",
        "low-frequency": 2500,
        "high-frequency": 10000,
        "low-amp": -120,
        "high-amp": -60,
        "top-freq": 12500,
    },
    {
        "target": "Sitophilus oryzae",
        "material": "Wheat Groats",
        "file": r"data/bug_bytes/A5-w40khzrw1.wav",
        "sensor": "ultrasonic (40kHz)",
        "low-frequency": 6000,
        "high-frequency": 9000,
        "low-amp": -120,
        "high-amp": -50,
        "top-freq": 12000,
    },
    {
        "target": "Sitophilus oryzae",
        "material": "Wheat Groats",
        "file": r"data/bug_bytes/A6-w30khzrw1.wav",
        "sensor": "ultrasonic (30kHz)",
        "low-frequency": 2000,
        "high-frequency": 6000,
        "low-amp": -120,
        "high-amp": -40,
        "top-freq": 25000,
    },
    {
        "target": "Sitophilus oryzae",
        "material": "Wheat Groats",
        "file": r"data/bug_bytes/A7-piezorw1.wav",
        "sensor": "piezoelectric",
        "low-frequency": 500,
        "high-frequency": 8000,
        "low-amp": -110,
        "high-amp": -50,
        "top-freq": 10000,
    },
    {
        "target": "Sitophilus zeamais larvae",
        "material": "Maize",
        "file": r"data/bug_bytes/A8-sz-lm3r1_f3-0s.wav",
        "sensor": "microphone",
        "low-frequency": 1000,
        "high-frequency": 4000,
        "low-amp": -120,
        "high-amp": -40,
        "top-freq": 10000,
    },
    {
        "target": "Sitophilus zeamais",
        "material": "Maize",
        "file": r"data/bug_bytes/A9-sza-mar1f5-0s.wav",
        "sensor": "microphone",
        "low-frequency": 1000,
        "high-frequency": 6000,
        "low-amp": -140,
        "high-amp": -60,
        "top-freq": 10000,
    },
    {
        "target": "Prostephanus trancatus larvae",
        "material": "Maize",
        "file": r"data/bug_bytes/A10-ptr-l-m4r2f4-2s.wav",
        "sensor": "microphone",
        "low-frequency": 1000,
        "high-frequency": 8000,
        "low-amp": -140,
        "high-amp": -60,
        "top-freq": 10000,
    },
    {
        "target": "Prostephanus trancatus",
        "material": "Maize",
        "file": r"data/bug_bytes/A11-ptr_adu1-72s.wav",
        "sensor": "microphone",
        "low-frequency": 1000,
        "high-frequency": 8000,
        "low-amp": -140,
        "high-amp": -60,
        "top-freq": 10000,
    },
]
targets = pd.DataFrame(targets)

# %%

figure_targets = [targets.iloc[1], targets.iloc[6]]
labels = ["13", "14"]

for i, t in enumerate(figure_targets):
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
        showscale="right",
        cmap=matplotlib.colormaps.get_cmap("cb.solstice"),
        # cmap=cb.cbmap("cb.solstice"),
    )
    ax.set_ylim(0, t["top-freq"])
    ax.set_yticks([0, round(t["top-freq"] / 2), t["top-freq"]])
    fig.savefig(
        rf"projects/Dissertation/dissertation/figures/{labels[i]}_spectrogram.pdf",
        dpi=300,
    )

    f, p = signal.welch(
        a.data.signal,
        fs=a.sample_rate,
        nperseg=1024,
        window="blackmanharris",
        scaling="spectrum",
        average="mean",
    )

    a.bandpass_filter(t["low-frequency"], t["high-frequency"], order=10, overwrite=True)
    a.envelope(overwrite=True)

    level = 1 * np.median(a.data.signal)
    noise = a.data[a.data.signal <= level]
    rms = np.sqrt(np.mean(noise.signal**2))

    f_n, p_n = signal.welch(
        noise.signal,
        fs=a.sample_rate,
        nperseg=1024,
        window="blackmanharris",
        scaling="spectrum",
        average="mean",
    )

    fig, ax = plt.subplots()
    ax.plot(
        f, 10 * np.log10(p), label="Signal", color=color.colors[0], linestyle="solid"
    )
    ax.plot(f_n, 10 * np.log10(p_n), label="Reference")
    ax.set_xlim(0, 8000)
    ax.set_ylim(-100, 0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectral Power [dB]", fontsize=8)
    ax.legend(ncols=2, loc="upper right")
    fig.savefig(
        rf"projects/Dissertation/dissertation/figures/{labels[i]}_spectra.pdf",
        dpi=300,
    )

    # fig, ax = plt.subplots()
    # ax.plot(f, 10 * np.log10(p) - 10 * np.log10(p_n))
    # ax.set_xlim(0, round(max(f)))
    # ax.set_ylim(None, None)
    # ax.set_xlabel("Frequency [Hz]")
    # ax.set_ylabel("SNR [dB]")
    # fig.savefig(
    #     rf"projects/Dissertation/proposal/figures/{t['sensor']} - {t['target']}_snr.pdf",
    #     dpi=300,
    # )

    p = p[(f >= t["low-frequency"]) & (f <= t["high-frequency"])]
    p = 10 * np.log10(p)
    spl = 10 * np.log10(np.sum(10 ** (p / 10)))

    p_n = 10 * np.log10(p_n)
    p_n = p_n[(f_n >= t["low-frequency"]) & (f_n <= t["high-frequency"])]
    npl = 10 * np.log10(np.sum(10 ** (p_n / 10)))

    snr = spl - npl
    print(f"{t['target']} is {snr} dB")

    targets.at[i, "NSEL"] = snr

    a.data.signal = a.data.signal / rms

    min_threshold = 0.5 * np.max(a.data.signal)
    max_threshold = 0.9 * np.max(a.data.signal)
    rms = np.sqrt(
        np.mean(
            a.data.signal[
                (a.data.signal >= min_threshold) & (a.data.signal <= max_threshold)
            ]
            ** 2
        )
    )

    targets.at[i, "NSPA"] = 20 * np.log10(rms)

    fig, ax = plt.subplots()
    ax.plot(a.data["time [s]"], a.data.signal)
    ax.hlines(
        [rms],
        0,
        round(a.data["time [s]"].max()),
        color="red",
        linestyle="dashed",
        label="NSPA ({:.0f} dB)".format(20 * np.log10(rms)),
    )
    # plot rectange at cutoffs
    ax.fill_between(
        a.data["time [s]"],
        min_threshold,
        max_threshold,
        color="red",
        alpha=0.25,
        label="Analysis Region",
    )
    ax.legend(ncols=2, loc="upper right")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized\nAmplitude")
    ax.set_ylim(0, None)
    ax.set_xlim(0, round(a.data["time [s]"].max()))
    fig.savefig(
        rf"projects/Dissertation/dissertation/figures/{labels[i]}_envelope.pdf",
        dpi=300,
    )
    # %%
    # fig.savefig(rf"projects/Dissertation/proposal/figures/bug_bytes/{t['sensor']} - {t['target']}_envelope.jpeg", dpi=300)

    # document.add_figure(doc, fig, ax, "Envelope", add_break=False)
    # doc.add_page_break()

# targets.NSPA = targets.NSPA.round(2)
# targets.NSEL = targets.NSEL.round(2)

# targets.to_csv(r"projects/Dissertation/proposal/figures/bug_bytes/2024_06_28_bug_bytes.csv", index=False)
# targets = targets.drop(columns=["file"])
# document.add_dataframe(doc, targets)
# doc.save(r"projects/Dissertation/proposal/figures/bug_bytes/2024_06_28_bug_bytes.docx")
# %%
# targets = pd.read_csv(r"projects/Dissertation/proposal/figures/bug_bytes/2024_06_28_bug_bytes.csv")
