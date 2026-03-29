# %%
from . import lookup
import numpy as np
import pandas as pd
from dankpy import color  # type: ignore
from matplotlib import pyplot as plt
from spidb import spidb

# %%
db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.latex")

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
    .filter(spidb.Sample.channel_id.in_([1, 2, 3, 4]))
    .filter(spidb.Sample.subject_id.in_([1, 2, 3, 4]))
    # .filter(spidb.Sample.noise == "Silence")
    .all()
)

s = pd.DataFrame([s.__dict__ for s in samples])
s["nsel 1565-6000"] = [
    sample.classifications[4].classification
    if len(sample.classifications) > 0
    else np.nan
    for sample in samples
]
s["nsel 1565-6000"] = s["nsel 1565-6000"].astype(float)
s["nsel 500-6000"] = [
    sample.classifications[4].classification
    if len(sample.classifications) > 1
    else np.nan
    for sample in samples
]
s["nsel 500-6000"] = s["nsel 500-6000"].astype(float)

s["spl 500-6000"] = [
    sample.classifications[4].classification
    if len(sample.classifications) > 2
    else np.nan
    for sample in samples
]
s["spl 500-6000"] = s["spl 500-6000"].astype(float)

s["spl 1565-6000"] = [
    sample.classifications[4].classification
    if len(sample.classifications) > 3
    else np.nan
    for sample in samples
]
s["spl 1565-6000"] = s["spl 1565-6000"].astype(float)
# %%
# classifications = db.session.query(spidb.Classification).filter(spidb.Classification.classifier == "nsel 1565-6000").filter(spidb.Subject.id.in_([1,2,3,4])).all()
# classes = pd.DataFrame([c.__dict__ for c in classifications])

# #%%
# classes = classes.drop(columns=["_sa_instance_state"])
# #%%
sub = (
    s[
        [
            "record_id",
            "subject_id",
            "material_id",
            "nsel 1565-6000",
            "nsel 500-6000",
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
# subdata = pd.read_csv(r"nsel-nsel_records.csv")
# %%
# good_targets = pd.read_pickle(
#     r"projects/MDPI-Detection/archive/bin/detection_results_2025-08-08.pkl"
# )
# good_targets = good_targets[["record_id", "internal_1", "external_1"]]
# good_targets["difference"] = good_targets["internal_1"] - good_targets["external_1"]
# good_targets = good_targets[good_targets.difference > 0]
# sub = sub[sub.record_id.isin(good_targets.record_id)]

sub["material"] = sub["material_id"].map(
    lambda x: db.session.get(spidb.Material, x).name
)
sub["subject"] = sub["subject_id"].map(lambda x: db.session.get(spidb.Subject, x).name)

sub["material"] = sub["material"].apply(
    lambda x: f"{lookup.lookup(x, latex=True, min=True)} \n {x}"
)
sub["subject"] = sub["subject"].apply(lambda x: lookup.lookup(x, latex=True, min=True))

noise_lvl = np.array([60, 70, 80, 90, 100])

materials = sub.material.unique()
materials.sort()

subjects = sub.subject.unique()
subjects.sort()

# %%
j = 0
band = "500-6000"
# band = "500-6000"
# subplot = sub[sub[f"snr {band}"] > 0]
subplot = sub

fig, ax = plt.subplots()
for m in materials:
    # for m in data.material.unique():
    group = subplot[subplot["material"] == m]
    group.sort_values(by="subject", inplace=True)

    artists = []
    for c, i in enumerate(subjects):
        # for i in data.target.unique():
        g = group[group["subject"] == i]

        values = g[f"nsel {band}"].values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        box_min = q1 - 5 * iqr
        box_max = q3 + 5 * iqr

        filtered_values = values[(values >= box_min) & (values <= box_max)]

        b = ax.boxplot(
            g[f"nsel {band}"].values,
            # filtered_values,
            positions=[j + 0.5],
            showfliers=False,
            boxprops={"facecolor": color.colors[c]},
            medianprops={"color": "black", "linewidth": 0},
            patch_artist=True,
            widths=0.25,
            showcaps=False,
            notch=False,
            whis=False,
            # whis=(0, 100),
        )
        j += 1
        artists.append(b)

    # p += 1
ax.set_ylim(0, None)

# place legend above the plot
ax.legend(
    [element["boxes"][0] for element in artists],
    list(group.subject.unique()),
    loc="upper center",
    ncols=4,
    columnspacing=1,
    markerscale=0.5,
    bbox_to_anchor=(0.5, 1.35),
    fontsize=10,
)

lims = ax.get_xlim()
ax.set_xticks(
    np.arange(0, lims[1], lims[1] // len(sub.material_id.unique()))
    + lims[1] // (2 * len(sub.material_id.unique()))
)
ax.vlines(np.arange(0, 20, 4), 0, 100, color="black", alpha=0.25)
ax.set_xticklabels(list(materials))
ax.vlines(np.arange(0, 20, 4), 0, 80, color="black", alpha=0.25)
ax.set_xticklabels(list(materials))
plt.tick_params(axis="x", which="both", bottom=False, top=False)

ax.set_ylabel("nsel [dB]")
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])

# remove the yticks marks from the right side
ax.yaxis.set_ticks_position("left")

fig.set_size_inches(6, 1.75)
fig.savefig(
    r"projects\Dissertation\dissertation\figures\4_12b_nsel_box.pdf",
    dpi=300,
    bbox_inches="tight",
)
# %%
