# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("dankpy.styles.stevens_presentation")


# data = pd.read_csv(r"projects/MDPI-Detection/scripts/nspa_results/nspa_grid.csv")
data = pd.read_csv(r"projects/MDPI-Detection/archive/data/nspa_grid.csv")
# %%
low_freq = np.linspace(1000, 5000, 41)
# high_freq = np.linspace(3000, 8000, 51)
# low_freq = np.linspace(500, 5000, 41)

high_freq = [6000]
# %%

# %%
# get the columns that only have nspa_number_number but not nspa_diff_number_number
nspa_columns = [
    col for col in data.columns if col.startswith("nspa_") and "diff" not in col
]

nspa = data[nspa_columns]
nspa = nspa.mean()

nspa_grid = []
for low in low_freq:
    for high in high_freq:
        key = f"nspa_sample_{int(low)}_{int(high)}"
        noise_key = f"nspa_noise_{int(low)}_{int(high)}"
        nspa_grid.append(
            {
                "low": low,
                "high": high,
                "nspa": nspa[key],
                "nspa_noise": nspa[noise_key],
            }
        )
nspa_grid = pd.DataFrame(nspa_grid)

grid = nspa_grid.pivot(index="low", columns="high", values="nspa")
# %%
# fig, ax = plt.subplots()
# im = ax.imshow(
#     grid,
#     aspect="auto",
#     origin="lower",
#     extent=[high_freq.min(), high_freq.max(), low_freq.min(), low_freq.max()],
#     vmin=20,
#     vmax=60,
#     cmap="jet",
# )
# cbar = fig.colorbar(im, ax=ax, label="NSPA [dB]")
# cbar.set_label(label="NSPA [dB]", size=9)
# # cbar.set_ticks([44, 48])
# ax.set_xlim(high_freq.min(), high_freq.max())
# # ax.set_xticks([4500, 5250, 6000])
# ax.set_ylim(low_freq.min(), low_freq.max())
# # ax.set_yticks([1500, 2250, 3000])

# # find the maximum nspa value
# max_val = grid.max().max()
# max_freq = grid.stack().idxmax()
# print(f"Maximum NSPA: {max_val} dB at {max_freq}")
# %%

nspa_columns = [col for col in data.columns if col.startswith("nspa_diff")]

nspa = data[nspa_columns]
nspa = nspa.mean()

nspa_grid = []
for low in low_freq:
    for high in high_freq:
        key = f"nspa_diff_{int(low)}_{int(high)}"
        nspa_grid.append(
            {
                "low": low,
                "high": high,
                "nspa": nspa[key],
            }
        )
nspa_grid = pd.DataFrame(nspa_grid)

grid = nspa_grid.pivot(index="low", columns="high", values="nspa")

# fig, ax = plt.subplots()
# im = ax.imshow(
#     grid,
#     aspect="auto",
#     origin="lower",
#     extent=[high_freq.min(), high_freq.max(), low_freq.min(), low_freq.max()],
#     vmin=0,
#     vmax=40,
#     cmap="jet",
# )
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(label="NSPA Difference [dB]")
# cbar.set_ticks([0, 20, 40])
# ax.set_xlim(high_freq.min(), high_freq.max())
# ax.set_ylim(low_freq.min(), low_freq.max())
# ax.set_yticks([500, 1500, 2500])

# # find the maximum difference
# max_val = grid.max().max()
# max_freq = grid.stack().idxmax()
# print(f"Maximum NSPA Difference: {max_val} dB at {max_freq}")

# ax.set_xlabel("High Frequency [Hz]")
# ax.set_ylabel("Low Frequency [Hz]")

# fig.savefig(r"projects/Dissertation/proposal/figures/nspa_optimization/nspa_diff_grid.pdf",dpi=300, bbox_inches="tight")

# %%
values = nspa_grid[nspa_grid["high"] == 6000]

nspa_low_freq = pd.read_csv("projects/MDPI-Detection/archive/data/nspa_grid_specific.csv")
average = nspa_low_freq.groupby("frequency").mean().reset_index()

fig, ax = plt.subplots(figsize=(5.63, 4.21))
ax.scatter(
    values[values.low < 1400]["low"], values[values.low < 1400]["nspa"], c="black", s=20
)
ax.scatter(
    values[values.low >= 1600]["low"],
    values[values.low >= 1600]["nspa"],
    c="black",
    s=20,
)


convergence = average[average["nspa_diff"].diff().abs() < 0.05].min()

ax.scatter(
    convergence["frequency"], convergence["nspa_diff"], color="red", s=20, zorder=3
)

average = average[average["frequency"] < 1550]  # remove the outlier at 1500
ax.scatter(
    average["frequency"].iloc[::5], average["nspa_diff"].iloc[::5], c="black", s=20
)

ax.set_xlim(1000, 5000)
ax.set_ylim(22, 42)
ax.set_yticks([22, 32, 42])
ax.set_xlabel("Lower Frequency ($f_l$) [Hz]")
ax.set_ylabel("NSPA Difference [dB]")
# fig.savefig(
#     r"projects\Dissertation\dissertation\figures\5_7_nspa_optimization.pdf",
#     dpi=300,
# )


# %%
