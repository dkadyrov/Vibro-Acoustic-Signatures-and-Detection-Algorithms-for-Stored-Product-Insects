# %%
from archive import lookup
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from dankpy import utilities, color, dt, dankframe, document, statistics
import multiprocessing as mp

from spidb import spidb, statistics
import matplotlib.pyplot as plt
#%%
db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

classifications = db.session.query(spid    b.Classification).filter(spidb.Classification.classifier == "nspa 500-6000").filter(spidb.Subject.id.in_([1,2,3,4])).all()
classes = pd.DataFrame([c.__dict__ for c in classifications])

#%%
classes = classes.drop(columns=["_sa_instance_state"])
#%%
classes = classes.groupby(["record_id"]).max().reset_index()
subdata = pd.read_csv(r"nspa-nsel_records.csv")

good_targets = pd.read_pickle(r"projects/MDPI-Detection/archive/bin/detection_results_2025-08-08.pkl")
good_targets = good_targets[["record_id", "internal_5", "external_5"]]
good_targets["difference"] = good_targets["internal_5"] - good_targets["external_5"]
good_targets = good_targets[good_targets.difference > 0]
# classes = classes[classes.record_id.isin(good_targets.record_id)]
#%%
# Count how many records in subdata are none
# print(f"Records in subdata that are None: {subdata.record_id.isna().sum()}")


#%%
# classes = classes[classes.record_id.isin(subdata.record)]
# %%
classes["material_id"] = classes["record_id"].map(lambda x: db.session.get(spidb.Record, x).material.id)

classes["target_id"] = classes["record_id"].map(lambda x: db.session.get(spidb.Record, x).subject.id)
# %%
classes = classes[classes["material_id"].isin([1,2,3,4,5])]
classes = classes[classes["target_id"].isin([1,2,3,4])]
#%%


classes["material"] = classes["material_id"].map(lambda x: db.session.get(spidb.Material, x).name)
classes["target"] = classes["target_id"].map(lambda x: db.session.get(spidb.Subject, x).name)

classes["material"] = classes["material"].apply(
    lambda x: f"{lookup.lookup(x, latex=True, min=True)} \n {x}"
)
classes["target"] = classes["target"].apply(
    lambda x: lookup.lookup(x, latex=True, min=True)
)#%%
classes["classification"] = classes["classification"].astype(float)

# %%
# external_noise = [{'measured': 60.0,
#   'NSPA (500-6000)': 43.356671232450246,
#   'NSPA (1565-6000)': 10.60558897927293},
#  {'measured': 70.0,
#   'NSPA (500-6000)': 47.302455096175244,
#   'NSPA (1565-6000)': 13.339319774699177},
#  {'measured': 80.0,
#   'NSPA (500-6000)': 51.881988540471546,
#   'NSPA (1565-6000)': 20.21534521985494},
#  {'measured': 90.0,
#   'NSPA (500-6000)': 58.49941297817654,
#   'NSPA (1565-6000)': 28.04559412963823},
#  {'measured': 100.0,
#   'NSPA (500-6000)': 61.55410442062532,
#   'NSPA (1565-6000)': 24.902243355841733}]
# external_noise = pd.DataFrame(external_noise)
external_noise = np.array([39.34808766, 46.86476964, 54.38145162, 61.8981336 , 69.41481558])
noise_lvl = np.array([60, 70, 80, 90, 100])
j = 0

materials = classes.material.unique()
materials.sort()

targets = classes.target.unique()
targets.sort()

fig, ax = plt.subplots()
for m in materials:
    # for m in data.material.unique():
    group = classes[classes["material"] == m]
    group.sort_values(by="target", inplace=True)

    artists = []
    c = 0
    for i in targets:
        # for i in data.target.unique():
        g = group[group["target"] == i]
        b = ax.boxplot(
            g.classification.values,
            positions=[j + 0.5],
            showfliers=False,
            boxprops=dict(facecolor=color.colors[c]),
            medianprops=dict(color="black", linewidth=0),
            patch_artist=True,
            widths=0.25,
            showcaps=True,
            notch=False,
            whis=True,
        )
        j += 1
        c += 1
        artists.append(b)
    # p += 1
ax.set_ylim(0, None)

# place legend above the plot
ax.legend(
    [element["boxes"][0] for element in artists],
    list(group.target.unique()),
    loc="upper center",
    ncols=4,
    columnspacing=1,
    markerscale=0.5,
    bbox_to_anchor=(0.5, 1.35),
)

lims = ax.get_xlim()
ax.set_xticks(
    np.arange(0, lims[1], lims[1] // len(classes.material_id.unique()))
    + lims[1] // (2 * len(classes.material_id.unique()))
)
ax.vlines(np.arange(0, 20, 4), 0, 100, color="black", alpha=0.25)
ax.set_xticklabels([material for material in materials])
ax.vlines(np.arange(0, 20, 4), 0, 80, color="black", alpha=0.25)
ax.set_xticklabels([material for material in materials])
plt.tick_params(axis="x", which="both", bottom=False, top=False)

for i, row in enumerate(external_noise):
    ax.axhline(external_noise[i], linestyle="--", label=f"{noise_lvl[i]} dB (1565-6000 Hz)", zorder=10)

    # make text on the right side of the line
    ax.text(20.25, external_noise[i], f"{noise_lvl[i]} dB", fontsize=8, color="black", zorder=10, va="center")

ax.set_ylabel("NSPA [dB]")
# ax.set_ylim(20, 80)
# ax.set_yticks([20, 50, 80])
ax.set_title("500-6000 Hz")
fig.set_size_inches(11.71, 4.24)
# %%
