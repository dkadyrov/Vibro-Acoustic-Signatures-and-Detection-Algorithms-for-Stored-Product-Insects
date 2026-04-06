# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from dankpy import utilities, colors, dt, dankframe, document, statistics
import multiprocessing as mp

from spidb import spidb, statistics
import matplotlib.pyplot as plt

db = spidb.Database(r"data/spi.db")

plt.style.use("dankpy.styles.stevens_presentation")

data = pd.read_pickle(r"data/statistics/combined/2024_06_11_combined.pkl")
noise = pd.read_pickle(r"data/noise_500-6000.pkl")

data["test-iteration"] = data["test"].astype(str) + "-" + data["iteration"].astype(str)
data["record"] = pd.factorize(data["test-iteration"])[0]
data = data.dropna()

data.target = data.target.str.replace("larvae", "larva")

noise_filt = noise[noise["frequency"] >= 500]
noise_filt = noise_filt[noise_filt["frequency"] <= 6000]
data.material = data.material.replace("Corn Meal", "Flour")

materials = ['Rice', 'Wheat Groats', "Flour", 'Corn Flakes', 'Oatmeal']
targets = data.target.unique()

data = data[data.material.isin(materials)]

for c in [0, 1, 2, 3, 7]:
    data = data.dropna(subset=[f"{c}_rms"])
    data = data.dropna(subset=[f"{c}_spl"])
    npl = 10*np.log10(np.sum(10**(10*np.log10(noise_filt[f"{c}"])/10)))
    data[f"{c}_snr"] = data[f"{c}_spl"] - npl

data["max_amp"] = data[[f"{c}_rms" for c in range(4)]].max(axis=1)
data["max_snr"] = data[[f"{c}_snr" for c in range(4)]].max(axis=1)

data = data[data["max_snr"] > 0]
data["max_amp"] = 20*np.log10(data["max_amp"])
data["factor"] = data["max_amp"] / data["max_snr"]
data["ratio"] = data["detection"] - data["noise"]

subdata = data.copy()
subdata = subdata[subdata.ratio > 0]
subdata = subdata[subdata.material.isin(["Oatmeal", "Rice", "Flour", "Wheat Groats", "Corn Flakes"])]
subdata = subdata.reset_index(drop=True)
#%%
# records = []
# for i, test in subdata.groupby("test"):
#     event = db.session.get(spidb.Event, i)
#     for j, record in test.iterrows():
#         start = event.start + dt.timedelta(seconds=60*record["iteration"])
#         r = db.session.query(spidb.Record).filter(spidb.Record.start == start).first()
#         if r is not None:
#             records.append(int(r.id))
#         else:
#             print(f"Record not found for test {i}, iteration {record['iteration']}")
#             records.append(None)
#             break
# subdata["record_id"] = records
# subdata.to_csv(r"nspa-nsel_records.csv")

#%%
fig, ax = statistics.generate_boxwhisker(subdata, "max_snr")
ax.set_ylabel("NSEL [dB]")
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])
fig.set_size_inches(11.71, 4.24)

# ax.hlines(37, 0, 20, color="black", linestyle="--", alpha=0.5, zorder=0)

# fig.savefig(r"projects/Dissertation/proposal/figures/nspa_nsel/nsel_box.pdf", dpi=300, bbox_inches="tight")

#%%
external_noise = [{'measured': 60.0,
  'NSPA (500-6000)': 43.356671232450246,
  'NSPA (1565-6000)': 10.60558897927293},
 {'measured': 70.0,
  'NSPA (500-6000)': 47.302455096175244,
  'NSPA (1565-6000)': 13.339319774699177},
 {'measured': 80.0,
  'NSPA (500-6000)': 51.881988540471546,
  'NSPA (1565-6000)': 20.21534521985494},
 {'measured': 90.0,
  'NSPA (500-6000)': 58.49941297817654,
  'NSPA (1565-6000)': 28.04559412963823},
 {'measured': 100.0,
  'NSPA (500-6000)': 61.55410442062532,
  'NSPA (1565-6000)': 24.902243355841733}]

external_noise = pd.DataFrame(external_noise)

fig, ax = statistics.generate_boxwhisker(subdata, "max_amp")

# plot horizontal lines with text on the right side
for i, row in external_noise.iterrows():
    ax.axhline(row["NSPA (500-6000)"], linestyle="--", label=f"{row['measured']} dB (500-6000 Hz)", zorder=10)

    # make text on the right side of the line
    ax.text(20.25, row["NSPA (500-6000)"], f"{row['measured']} dB", fontsize=8, color="black", zorder=10, va="center")



ax.set_ylabel("NSPA [dB]")
ax.set_ylim(20, 80)
ax.set_yticks([20, 50, 80])
ax.set_title("500-6000 Hz")
fig.set_size_inches(11.71, 4.24)

# fig.savefig(r"projects/Dissertation/proposal/figures/nspa_nsel/nspa_box.pdf", dpi=300, bbox_inches="tight")

#%%

# fig, ax = statistics.generate_boxwhisker(subdata, "detection")
# ax.set_ylabel("Number of Impulses")

# #%%

# fig, ax = statistics.generate_boxwhisker(subdata, "factor")
# ax.set_ylabel("Factor [RMS [dB]/SNR [dB]]")
# #%%
# snr = pd.pivot_table(data, index="target", columns="material", values="max_snr", aggfunc=np.mean)
# snr.round(2)
# #%%
# rms = pd.pivot_table(data, index="target", columns="material", values=["max_amp", "max_snr"], aggfunc=np.mean)
# rms.round(2)
# #%%
# #create a table from the data dataframe that contains the number of unique records and tests for each target and material
# records = pd.pivot_table(data, index="target", columns="material", values="record", aggfunc="count")
# records
# #%%
# tests = pd.pivot_table(data, index="target", columns="material", values="test", aggfunc="nunique")
# tests

# #%%
# var = pd.pivot_table(data, index="target", columns="material", values="max_amp", aggfunc=np.var)
# var.round(2)

# #%%
amp = pd.pivot_table(data, index="target", columns="material", values="max_amp", aggfunc=np.mean)
# amp.round(2)
# # %%
def closest_to_medium(values):
    medium = np.median(values)
    return values.index[np.argmin(np.abs(values - medium))]

def closest_to_mean(values):
    mean = np.mean(values)
    return values.index[np.argmin(np.abs(values - mean))]

rows = pd.pivot_table(subdata, index="target", columns="material", values="max_amp", aggfunc=closest_to_mean)

# #%%
# oatmeal = subdata[(subdata.material == "Oatmeal") & (subdata.target == "Callosobruchus maculatus")]
# # %%
