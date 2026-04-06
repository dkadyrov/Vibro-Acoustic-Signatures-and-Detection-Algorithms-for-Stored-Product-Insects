#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dankpy

plt.style.use("dankpy.styles.stevens_presentation")


data = pd.read_csv(r"projects/MDPI-Detection/scripts/nspa_results/nspa_grid.csv")
# %%
low_freq = np.linspace(500, 2500, 21)
high_freq = np.linspace(3000, 8000, 51)

#%%

# %%
# get the columns that only have nspa_number_number but not nspa_diff_number_number
nspa_columns = [col for col in data.columns if col.startswith("nspa_") and "diff" not in col]

nspa = data[nspa_columns]
nspa = nspa.mean()

nspa_grid = [] 
for low in low_freq:
    for high in high_freq:
        key = f"nspa_{int(low)}_{int(high)}"
        nspa_grid.append({
            "low": low,
            "high": high,
            "nspa": nspa[key],
        })
nspa_grid = pd.DataFrame(nspa_grid)

grid = nspa_grid.pivot(index="low", columns="high", values="nspa")

fig, ax = plt.subplots()
im = ax.imshow(grid, aspect="auto", origin="lower", extent=[high_freq.min(), high_freq.max(), low_freq.min(), low_freq.max()], vmin=20, vmax=60,cmap="jet")
cbar = fig.colorbar(im, ax=ax, label="NSPA [dB]")
cbar.set_label(label='NSPA [dB]', size=9)
# cbar.set_ticks([44, 48])
ax.set_xlim(high_freq.min(), high_freq.max())
# ax.set_xticks([4500, 5250, 6000])
ax.set_ylim(low_freq.min(), low_freq.max())
# ax.set_yticks([1500, 2250, 3000])

# find the maximum nspa value
max_val = grid.max().max()
max_freq = grid.stack().idxmax()
print(f"Maximum NSPA: {max_val} dB at {max_freq}")
# %%

nspa_columns = [col for col in data.columns if col.startswith("nspa_diff")]

nspa = data[nspa_columns]
nspa = nspa.mean()

nspa_grid = [] 
for low in low_freq:
    for high in high_freq:
        key = f"nspa_diff_{int(low)}_{int(high)}"
        nspa_grid.append({
            "low": low,
            "high": high,
            "nspa": nspa[key],
        })
nspa_grid = pd.DataFrame(nspa_grid)

grid = nspa_grid.pivot(index="low", columns="high", values="nspa")

fig, ax = plt.subplots()
im = ax.imshow(grid, aspect="auto", origin="lower", extent=[high_freq.min(), high_freq.max(), low_freq.min(), low_freq.max()], vmin=0, vmax=40,cmap="jet")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label='NSPA Difference [dB]')
cbar.set_ticks([0, 20, 40])
ax.set_xlim(high_freq.min(), high_freq.max())
ax.set_ylim(low_freq.min(), low_freq.max())
ax.set_yticks([500, 1500, 2500])

# find the maximum difference
max_val = grid.max().max()
max_freq = grid.stack().idxmax()
print(f"Maximum NSPA Difference: {max_val} dB at {max_freq}")

ax.set_xlabel("High Frequency [Hz]")
ax.set_ylabel("Low Frequency [Hz]")

# fig.savefig(r"projects/Dissertation/proposal/figures/nspa_optimization/nspa_diff_grid.pdf",dpi=300, bbox_inches="tight")

# %%
values = nspa_grid[nspa_grid["high"] == 6000]

nspa_low_freq = pd.read_csv("projects/MDPI-Detection/scripts/nspa_results/nspa_grid_specific.csv")
average = nspa_low_freq.groupby("frequency").mean().reset_index()
#%%
fig, ax = plt.subplots()
ax.scatter(values["low"], values["nspa"], c="black", s=30)

#plot every 10th point of average
# ax.scatter(average["frequency"], average["nspa_diff"], c="black", s=2)
ax.scatter(average["frequency"].iloc[::5], average["nspa_diff"].iloc[::5], c="black", s=20)


convergence = average[average["nspa_diff"].diff().abs() < 0.05].min()

# ...existing code...
# Make the red dot slightly bigger
ax.scatter(convergence["frequency"], convergence["nspa_diff"], color="red", s=30, zorder=3)

# Add a black circle outline
# ax.scatter(convergence["frequency"], convergence["nspa_diff"], 
        #   facecolors='none', edgecolors='red', s=80, linewidths=2, zorder=2)

ax.set_xlim(500, 2500)
ax.set_ylim(0, 60)
ax.set_xlabel("Lower Frequency, $\\omega_1$ [Hz]")
ax.set_ylabel("NSPA Difference [dB]")
fig.set_size_inches(5.67, 4.21)
# fig.savefig(r"projects/Dissertation/proposal/figures/nspa_optimization/nspa_low_frequency.pdf",dpi=300, bbox_inches="tight")
# %%
