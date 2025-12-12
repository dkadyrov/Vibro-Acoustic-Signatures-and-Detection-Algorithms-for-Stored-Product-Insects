from archive import lookup
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# from archive.scripts.mdpi.drafts import external_noise
from dankpy import utilities, color, dt, dankframe, document, statistics
import multiprocessing as mp

from spidb import spidb, statistics
import matplotlib.pyplot as plt

# %%
db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
noise_filt_500 = noise[noise["frequency"] >= 500]
noise_filt_500 = noise_filt_500[noise_filt_500["frequency"] <= 6000]

noise_filt_1565 = noise[noise["frequency"] >= 1565]
noise_filt_1565 = noise_filt_1565[noise_filt_1565["frequency"] <= 6000]

npl_500 = 10 * np.log10(np.sum(10 ** (10 * np.log10(noise_filt_500["0"]) / 10)))
npl_1565 = 10 * np.log10(np.sum(10 ** (10 * np.log10(noise_filt_1565["0"]) / 10)))

samples = (
    db.session.query(spidb.Sample)
    .filter(spidb.Sample.sensor_id == 1)
    .filter(spidb.Sample.channel_id.in_([1, 2, 3, 4, 6]))
    .filter(spidb.Sample.subject_id.in_([1, 2, 3, 4, 6]))
    .all()
)

s = pd.DataFrame([s.__dict__ for s in samples])
s["nspa 1565-6000"] = [
    sample.classifications[0].classification
    if len(sample.classifications) > 0
    else np.nan
    for sample in samples
]
s["nspa 1565-6000"] = s["nspa 1565-6000"].astype(float)
s["nspa 500-6000"] = [
    sample.classifications[1].classification
    if len(sample.classifications) > 1
    else np.nan
    for sample in samples
]
s["nspa 500-6000"] = s["nspa 500-6000"].astype(float)

s["spl 500-6000"] = [
    sample.classifications[2].classification
    if len(sample.classifications) > 2
    else np.nan
    for sample in samples
]
s["spl 500-6000"] = s["spl 500-6000"].astype(float)

s["spl 1565-6000"] = [
    sample.classifications[3].classification
    if len(sample.classifications) > 3
    else np.nan
    for sample in samples
]
s["spl 1565-6000"] = s["spl 1565-6000"].astype(float)
# %%
# classifications = db.session.query(spidb.Classification).filter(spidb.Classification.classifier == "nspa 1565-6000").filter(spidb.Subject.id.in_([1,2,3,4])).all()
# classes = pd.DataFrame([c.__dict__ for c in classifications])

# #%%
# classes = classes.drop(columns=["_sa_instance_state"])
# #%%
internal = s[s.channel_id.isin([1, 2, 3, 4])]
external = s[s.channel_id == 6]

sub = (
    internal[
        [
            "record_id",
            "subject_id",
            "material_id",
            "nspa 1565-6000",
            "nspa 500-6000",
            "spl 500-6000",
            "spl 1565-6000",
        ]
    ]
    .groupby(["record_id"])
    .max()
    .reset_index()
)

mic = (
    external[
        [
            "record_id",
            "nspa 1565-6000",
            "nspa 500-6000",
            "spl 500-6000",
            "spl 1565-6000",
        ]
    ]
    .groupby(["record_id"])
    .max()
    .reset_index()
)

sub["snr 500-6000"] = sub["spl 500-6000"] - npl_500
sub["snr 1565-6000"] = sub["spl 1565-6000"] - npl_1565
# sub = s.groupby(["record_id"]).max().reset_index()
# subdata = pd.read_csv(r"nspa-nsel_records.csv")
# %%
# good_targets = pd.read_pickle(r"projects/MDPI-Detection/archive/bin/detection_results_2025-08-08.pkl")
# good_targets = good_targets[["record_id", "internal_1", "external_1"]]
# good_targets["difference"] = good_targets["internal_1"] - good_targets["external_1"]
# good_targets = good_targets[good_targets.difference > 0]
# # # sub = sub[sub.record_id.isin(good_targets.record_id)]

# sub["material"] = sub["material_id"].map(lambda x: db.session.get(spidb.Material, x).name)
# sub["subject"] = sub["subject_id"].map(lambda x: db.session.get(spidb.Subject, x).name)


# sub["material"] = sub["material"].apply(
# lambda x: f"{lookup.lookup(x, latex=True, min=True)} \n {x}"
# )
# sub["subject"] = sub["subject"].apply(
# lambda x: lookup.lookup(x, latex=True, min=True)
# )
# %%
band = "nspa 1565-6000"

records = pd.DataFrame()
records["record_id"] = sub["record_id"]
records["subject_id"] = sub["subject_id"]
records["material_id"] = sub["material_id"]
records["snr"] = sub["snr 1565-6000"]
records["Ch. 0 NSPA"] = records["record_id"].apply(
    lambda x: s[(s.record_id == x) & (s.channel_id == 1)][band].iloc[0]
)
records["Ch. 1 NSPA"] = records["record_id"].apply(
    lambda x: s[(s.record_id == x) & (s.channel_id == 2)][band].iloc[0]
)
records["Ch. 2 NSPA"] = records["record_id"].apply(
    lambda x: s[(s.record_id == x) & (s.channel_id == 3)][band].iloc[0]
)
records["Ch. 3 NSPA"] = records["record_id"].apply(
    lambda x: s[(s.record_id == x) & (s.channel_id == 4)][band].iloc[0]
)
records["Ch. 7 NSPA"] = records["record_id"].apply(
    lambda x: mic[mic.record_id == x][band].iloc[0]
)
records["Ch. 7 NSPA 500-6000"] = records["record_id"].apply(
    lambda x: mic[mic.record_id == x]["nspa 500-6000"].iloc[0]
)
records["Max NSPA"] = records[
    ["Ch. 0 NSPA", "Ch. 1 NSPA", "Ch. 2 NSPA", "Ch. 3 NSPA"]
].max(axis=1)
records["Max Channel"] = records[
    ["Ch. 0 NSPA", "Ch. 1 NSPA", "Ch. 2 NSPA", "Ch. 3 NSPA"]
].idxmax(axis=1)
records["External SPL"] = records.record_id.apply(
    lambda x: db.session.get(spidb.Record, x).external_spl
)
# Calculate how similar the channels are by finding the standard deviation of the channels
# records["Channel Std"] = records[["Ch. 0 NSPA", "Ch. 1 NSPA", "Ch. 2 NSPA", "Ch. 3 NSPA"]].std(axis=1)
records["Channel Std"] = records[["Max NSPA", "Ch. 7 NSPA"]].std(axis=1)
# %%
thresholds = np.linspace(0, 100, 100)

pod = []
pfp = []
pfn = []

problems = [435, 438, 439, 443, 456, 458, 459, 471, 479, 492, 579, 580, 581, 582, 583]
problems.append(np.arange(584, 597).tolist())

for t in thresholds:
    subrecords = records.copy()

    subrecords["Ch. 0 Detect"] = subrecords["Ch. 0 NSPA"] >= t
    subrecords["Ch. 1 Detect"] = subrecords["Ch. 1 NSPA"] >= t
    subrecords["Ch. 2 Detect"] = subrecords["Ch. 2 NSPA"] >= t
    subrecords["Ch. 3 Detect"] = subrecords["Ch. 3 NSPA"] >= t

    # count how many channels detect
    subrecords["Detections"] = subrecords[
        ["Ch. 0 Detect", "Ch. 1 Detect", "Ch. 2 Detect", "Ch. 3 Detect"]
    ].sum(axis=1)
    # if the number of detections is 1 or 2 then it is a detection, if its 0 then it is a non-detection, if its 3 or 4 then it is noise
    # subrecords = subrecords[subrecords["Max NSPA"] <= 80]
    # subrecords = subrecords[subrecords["Channel Std"] < 15]
    # subrecords = subrecords[subrecords["Ch. 7 NSPA 500-6000"] < 10]

    subrecords["Result"] = subrecords["Detections"].apply(
        lambda x: "Detection"
        if x in [1, 2]
        else ("Non-Detection" if x == 0 else "Noise")
    )

    # drop the rows that are noise
    subrecords = subrecords[subrecords.Result != "Noise"]
    subrecords = subrecords[subrecords.material_id.isin([1, 2, 3, 4, 5])]
    subrecords = subrecords[~subrecords.record_id.isin(problems)]


    insect = subrecords[subrecords.subject_id.isin([1, 2, 3, 4])]
    no_insect = subrecords[subrecords.subject_id == 6]

    tp = np.sum(insect["Result"] == "Detection")
    fn = np.sum(insect["Result"] == "Non-Detection")
    fp = np.sum(no_insect["Result"] == "Detection")

    pod.append(tp / len(insect) * 100)
    pfp.append(fp / len(no_insect) * 100)
    pfn.append(fn / len(insect) * 100)

results = pd.DataFrame({"threshold": thresholds, "pod": pod, "pfp": pfp, "pfn": pfn})
results["pod - pfp"] = results["pod"] - results["pfp"]
results["pod - pfn"] = results["pod"] - results["pfn"]
results["min diff"] = results[["pod - pfp", "pod - pfn"]].min(axis=1)
best = results.iloc[results["min diff"].idxmax()]

fig, ax = plt.subplots()
ax.plot(results.threshold, results.pod, label="POD", color=color.colors[0])
ax.plot(results.threshold, results.pfp, label="PFP", color=color.colors[1])
ax.plot(results.threshold, results.pfn, label="PFN", color=color.colors[2])
ax.axvline(
    best.threshold,
    color="gray",
    linestyle="--",
    label=f"Best Threshold: {best.threshold:.1f} dB NSPA\nPOD: {best.pod:.1f}%, PFP: {best.pfp:.1f}%, PFN: {best.pfn:.1f}%",
)

ax.set_xlabel("Detection Threshold [dB NSPA]")
ax.set_ylabel("Probability")
ax.legend(loc="upper right")
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)

# find at what row in results is the maximum pod with the maximum difference between pod and pfp as well as pod and pfn
print(best)
# %%
