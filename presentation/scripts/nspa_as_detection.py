from . import lookup
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
# from archive.scripts.mdpi.drafts import external_noise
from dankpy import utilities, color, dt, dankframe, document, statistics
import multiprocessing as mp

from spidb import spidb, statistics
import matplotlib.pyplot as plt
#%%
db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

noise = pd.read_pickle(r"data/noise_500-6000.pkl")
noise_filt_500 = noise[noise["frequency"] >= 500]
noise_filt_500 = noise_filt_500[noise_filt_500["frequency"] <= 6000]

noise_filt_1565 = noise[noise["frequency"] >= 1565]
noise_filt_1565 = noise_filt_1565[noise_filt_1565["frequency"] <= 6000]

npl_500 = 10*np.log10(np.sum(10**(10*np.log10(noise_filt_500["0"])/10)))
npl_1565 = 10*np.log10(np.sum(10**(10*np.log10(noise_filt_1565["0"])/10)))

samples = db.session.query(spidb.Sample).filter(spidb.Sample.sensor_id == 1).filter(spidb.Sample.channel_id.in_([1,2,3,4,6])).filter(spidb.Sample.subject_id.in_([1,2,3,4,6])).all()

s = pd.DataFrame([s.__dict__ for s in samples])
s["nspa 1565-6000"] = [sample.classifications[0].classification if len(sample.classifications) > 0 else np.nan for sample in samples]
s["nspa 1565-6000"] = s["nspa 1565-6000"].astype(float)
s["nspa 500-6000"] = [sample.classifications[1].classification if len(sample.classifications) > 1 else np.nan for sample in samples]
s["nspa 500-6000"] = s["nspa 500-6000"].astype(float)

s["spl 500-6000"] = [sample.classifications[2].classification if len(sample.classifications) > 2 else np.nan for sample in samples]
s["spl 500-6000"] = s["spl 500-6000"].astype(float)

s["spl 1565-6000"] = [sample.classifications[3].classification if len(sample.classifications) > 3 else np.nan for sample in samples]
s["spl 1565-6000"] = s["spl 1565-6000"].astype(float)
#%%
# classifications = db.session.query(spidb.Classification).filter(spidb.Classification.classifier == "nspa 1565-6000").filter(spidb.Subject.id.in_([1,2,3,4])).all()
# classes = pd.DataFrame([c.__dict__ for c in classifications])

# #%%
# classes = classes.drop(columns=["_sa_instance_state"])
# #%%
internal = s[s.channel_id.isin([1,2,3,4])]
external = s[s.channel_id == 6]

sub = internal[["record_id", "subject_id", "material_id", "nspa 1565-6000", "nspa 500-6000", "spl 500-6000", "spl 1565-6000"]].groupby(["record_id"]).max().reset_index()

mic = external[["record_id", "nspa 1565-6000", "nspa 500-6000", "spl 500-6000", "spl 1565-6000"]].groupby(["record_id"]).max().reset_index()

sub["snr 500-6000"] = sub["spl 500-6000"] - npl_500
sub["snr 1565-6000"] = sub["spl 1565-6000"] - npl_1565

# sub = s.groupby(["record_id"]).max().reset_index()
# subdata = pd.read_csv(r"nspa-nsel_records.csv")
#%%
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
records = pd.DataFrame()
records["record_id"] = sub["record_id"]
records["subject_id"] = sub["subject_id"]
records["material_id"] = sub["material_id"]
records["Ch. 0 NSPA"] = records["record_id"].apply(lambda x: s[(s.record_id == x) & (s.channel_id ==1)]["nspa 1565-6000"].iloc[0])
records["Ch. 1 NSPA"] = records["record_id"].apply(lambda x: s[(s.record_id == x) & (s.channel_id ==2)]["nspa 1565-6000"].iloc[0])
records["Ch. 2 NSPA"] = records["record_id"].apply(lambda x:
    s[(s.record_id == x) & (s.channel_id ==3)]["nspa 1565-6000"].iloc[0])
records["Ch. 3 NSPA"] = records["record_id"].apply(lambda x: s[(s.record_id == x) & (s.channel_id ==4)]["nspa 1565-6000"].iloc[0])
#%%
band = "nspa 1565-6000"

# s = sub[sub["snr 1565-6000"] > 0]

insect = s[s.subject_id.isin([1,2,3,4])]
no_insect = s[s.subject_id == 6]


pod = [] 
pfp = []
pfn = [] 


thresholds = np.linspace(0, 100, 100)
for t in thresholds:
    tp = np.sum((insect[band] >= t))

    pod.append(tp /len(insect)*100)
    pfp.append(np.sum((no_insect[band] >= t)) / len(no_insect)*100)
    pfn.append(np.sum((insect[band] < t)) / len(insect)*100)

results = pd.DataFrame({"threshold": thresholds, "pod": pod, "pfp": pfp, "pfn": pfn})
fig, ax = plt.subplots()
ax.plot(results.threshold, results.pod, label="POD", color=color.colors[0])
ax.plot(results.threshold, results.pfp, label="PFP", color=color.colors[1])
ax.plot(results.threshold, results.pfn, label="PFN", color=color.colors[2])
ax.set_xlabel("Detection Threshold [dB NSPA]")
ax.set_ylabel("Probability")
ax.legend(loc="upper right")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# %%
