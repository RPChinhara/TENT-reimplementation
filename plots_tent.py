import re
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "./output"
SAVE_DIR = "./output/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]

METHODS = ["source", "norm", "tent"]

# -------- GET LATEST FILE --------
def get_latest(pattern):
    files = glob.glob(os.path.join(LOG_DIR, pattern))
    return max(files, key=os.path.getmtime)

FILES = {m: get_latest(f"{m}_*.txt") for m in METHODS}

print("Using logs:")
for k, v in FILES.items():
    print(k, "->", v)

# -------- PARSE --------
def parse(filepath):
    data = {c: [None]*5 for c in CORRUPTIONS}

    with open(filepath) as f:
        for line in f:
            match = re.search(r'error % \[(\w+?)(\d)\]: ([\d.]+)%', line)
            if match:
                c = match.group(1)
                s = int(match.group(2)) - 1
                val = float(match.group(3))
                if c in data:
                    data[c][s] = val
    return data

all_data = {m: parse(p) for m, p in FILES.items()}

# -------- CSV --------
rows = []
for m, d in all_data.items():
    sev5 = [d[c][4] for c in CORRUPTIONS]
    mean = np.mean(sev5)
    rows.append([m, mean] + sev5)

df = pd.DataFrame(rows, columns=["method", "mean"] + CORRUPTIONS)
df.to_csv("results.csv", index=False)
print("✅ CSV saved")

# -------- BAR PLOT (severity 5) --------
plt.figure(figsize=(10,5))
for m in METHODS:
    vals = [all_data[m][c][4] for c in CORRUPTIONS]
    plt.plot(vals, label=m)

plt.xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=45)
plt.ylabel("Error (%)")
plt.title("Severity 5 Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/1_severity5_bar.png")
plt.close()

# -------- TREND PLOT --------
plt.figure()
for m in METHODS:
    vals = []
    for s in range(5):
        tmp = [all_data[m][c][s] for c in CORRUPTIONS]
        vals.append(np.mean(tmp))
    plt.plot(range(1,6), vals, marker='o', label=m)

plt.xlabel("Severity")
plt.ylabel("Mean Error (%)")
plt.title("Severity Trend")
plt.legend()
plt.savefig(f"{SAVE_DIR}/2_severity_trend.png")
plt.close()

# -------- HEATMAP --------
heat = []
for m in METHODS:
    heat.append([all_data[m][c][4] for c in CORRUPTIONS])

heat = np.array(heat)

plt.imshow(heat)
plt.colorbar()
plt.xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=45)
plt.yticks(range(len(METHODS)), METHODS)
plt.title("Heatmap (Severity 5)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/3_heatmap.png")
plt.close()

# -------- SUMMARY TABLE IMAGE --------
fig, ax = plt.subplots()
ax.axis('off')
tbl = ax.table(cellText=df.round(2).values,
               colLabels=df.columns,
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)

plt.savefig(f"{SAVE_DIR}/4_summary_table.png")
plt.close()

print("🔥 All plots saved in output/plots/")