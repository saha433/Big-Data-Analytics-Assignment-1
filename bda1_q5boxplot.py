"""
plot_boxplots.py

Reads the CSV files written by stream_stats.c and generates three figures:
  box_60min.png  (or box_Xs.png)  — overall dataset
  box_10min.png                   — one box per 10-minute block
  box_1min.png                    — one box per 1-minute interval

Adapts automatically to whatever duration was used in the simulation
by reading /tmp/sim_config.txt.

Usage:
    python plot_boxplots.py
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Read simulation config ────────────────────────────────────────────────────
cfg = {}
with open("/tmp/sim_config.txt") as f:
    for line in f:
        k, v = line.strip().split("=")
        cfg[k] = int(v) if v.isdigit() else v

total_seconds  = int(cfg["total_seconds"])
total_minutes  = int(cfg["total_minutes"])
ten_min_blocks = int(cfg["ten_min_blocks"])
total_values   = int(cfg["total_values"])

def hms(s):
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    parts = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s: parts.append(f"{s}s")
    return " ".join(parts) if parts else "0s"

duration_label = hms(total_seconds)

# ── CSV reader ────────────────────────────────────────────────────────────────
def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {}
            for k, v in row.items():
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows

# ── Draw boxes from 5-number summary ─────────────────────────────────────────
def draw_boxes(ax, positions, rows, color, width=0.6):
    box_data = [{
        "med":    r["median"],
        "q1":     r["p25"],
        "q3":     r["p75"],
        "whislo": r["min"],
        "whishi": r["max"],
        "fliers": [],
    } for r in rows]

    ax.bxp(box_data,
           positions=positions,
           widths=width,
           showfliers=False,
           showmeans=False,
           patch_artist=True,
           medianprops=dict(color="white", linewidth=2),
           boxprops=dict(facecolor=color, color=color, alpha=0.82),
           whiskerprops=dict(color=color, linewidth=1.2, linestyle="--"),
           capprops=dict(color=color, linewidth=1.5))

    # Mean diamond overlay
    for pos, r in zip(positions, rows):
        ax.plot(pos, r["mean"], marker="D", color="#FAAD14",
                markersize=5, zorder=5)

def style_ax(ax, title, xlabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Value (uniform [0, 1])", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        mpatches.Patch(alpha=0.82, label="IQR box (Q1–Q3)"),
        plt.Line2D([0], [0], color="white", lw=2,   label="Median"),
        plt.Line2D([0], [0], marker="D", color="#FAAD14",
                   markersize=5, linestyle="None",   label="Mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=8,
              framealpha=0.7, loc="lower right")

# ── 1. Overall (full duration) box plot ──────────────────────────────────────
global_rows = read_csv("/tmp/global_stats.csv")
overall = [r for r in global_rows if r["interval"] == "Overall"]

fig, ax = plt.subplots(figsize=(4, 6))
style_ax(ax,
         title=f"Box plot — full dataset\n({duration_label}, {total_values:,} values)",
         xlabel="Dataset")
draw_boxes(ax, [1], overall, color="#5B8FF9")
ax.set_xticks([1])
ax.set_xticklabels([f"{duration_label}\nOverall"])
plt.tight_layout()
out1 = f"box_{total_seconds}s_overall.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")

# ── 2. 10-minute block box plots ──────────────────────────────────────────────
ten_rows = read_csv("/tmp/ten_min_stats.csv")
for r in ten_rows:
    r["block_num"] = int(r["block"])

positions = [r["block_num"] for r in ten_rows]

def block_label(b):
    s = (b - 1) * 10 + 1
    e = min(b * 10, total_minutes)
    return f"m{s}–{e}"

labels = [block_label(b) for b in positions]

fig_w = max(6, len(positions) * 1.4)
fig, ax = plt.subplots(figsize=(fig_w, 6))
style_ax(ax,
         title=f"Box plots — 10-minute blocks ({len(positions)} blocks)",
         xlabel="10-minute block")
draw_boxes(ax, positions, ten_rows, color="#5AD8A6",
           width=max(0.3, 0.7 - 0.03 * len(positions)))
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
plt.tight_layout()
out2 = "box_10min_blocks.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")

# ── 3. Per-minute box plots ───────────────────────────────────────────────────
min_rows = read_csv("/tmp/minute_stats.csv")
for r in min_rows:
    r["minute_num"] = int(r["minute"])

positions = [r["minute_num"] for r in min_rows]

fig_w = max(10, len(positions) * 0.32)
fig, ax = plt.subplots(figsize=(fig_w, 6))
style_ax(ax,
         title=f"Box plots — 1-minute intervals ({len(positions)} minutes)",
         xlabel="Minute")
draw_boxes(ax, positions, min_rows, color="#FF99C3",
           width=max(0.2, 0.6 - 0.005 * len(positions)))

# Tick every 5 minutes (or every minute if short run)
step = 1 if total_minutes <= 20 else 5
ax.set_xticks(positions[::step])
ax.set_xticklabels([str(p) for p in positions[::step]], fontsize=8)
plt.tight_layout()
out3 = "box_1min_intervals.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")

print(f"\nAll 3 box plots generated for a {duration_label} simulation.")
print("Expected values for U(0,1):")
print("  Mean ≈ Median ≈ 0.5000")
print("  Q1 ≈ 0.2500,  Q3 ≈ 0.7500,  IQR ≈ 0.5000")
print("  No outliers (Tukey fences lie outside [0, 1])")
