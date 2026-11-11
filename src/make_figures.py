"""Summary tables (Markdown) and figures from results/*.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

full = json.load(open("results/full.json"))
few = json.load(open("results/fewshot.json"))
logs = json.load(open("results/logs_bgl.json"))
PAPER_5 = {"KinyarwandaNews": 0.458, "KirundiNews": 0.541, "DengueFilipino": 0.652, "SwahiliNews": 0.627}
lines = ["## Table A. Full training set", "",
         "| Dataset | Paper | Ours k=2 optimistic | Top-2 contains | k=1 | k=2 random | k=2 nearest | TF-IDF+LR | Dup rate | k=1 clean | LR clean |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for d, r in full.items():
    lines.append(f"| {d} (n={r['n_test']}) | {r['paper']:.3f} | {r['k2_optimistic']:.3f} | {r['top2_contains']:.3f} | "
                 f"{r['k1_random']:.3f} | {r['k2_random']:.3f}±{r['k2_random_std']:.3f} | {r['k2_nearest']:.3f} | "
                 f"{r['tfidf_lr']:.3f} | {r['dup_rate']:.3f} | {r['k1_acc_on_clean']:.3f} | {r['tfidf_lr_on_clean']:.3f} |")
lines += ["", "## Table B. k sensitivity (random tie-break, mean of 5 seeds)", "",
          "| Dataset | k=1 | k=2 | k=3 | k=5 | k=7 | k=3 opt | k=5 opt |", "|---|---|---|---|---|---|---|---|"]
for d, r in full.items():
    lines.append(f"| {d} | " + " | ".join(f"{r[f'k{k}_random']:.3f}" for k in (1, 2, 3, 5, 7))
                 + f" | {r['k3_optimistic']:.3f} | {r['k5_optimistic']:.3f} |")
lines += ["", "## Table C. 5-shot (mean ± 95% CI over 5 trials)", "",
          "| Dataset | Paper gzip | Ours k=2 optimistic | k=1 | k=2 random | k=2 nearest | TF-IDF+LR |", "|---|---|---|---|---|---|---|"]
def mci(v):
    v = np.array(v); return f"{v.mean():.3f}±{1.96*v.std(ddof=1)/np.sqrt(len(v)):.3f}"
for d, runs in few.items():
    g = lambda key: [r[key] for r in runs]
    lines.append(f"| {d} | {PAPER_5[d]:.3f} | {mci(g('k2_optimistic'))} | {mci(g('k1_random'))} | "
                 f"{mci(g('k2_random'))} | {mci(g('k2_nearest'))} | {mci(g('tfidf_lr'))} |")
lines += ["", "## Table D. BGL_2k alert detection (mean over 5 splits)", "",
          "| Split | majority acc | gzip k=1 F1 | gzip k=2 opt F1 | gzip k=2 rand F1 | TF-IDF+LR F1 |", "|---|---|---|---|---|---|"]
for s in ("random", "template_disjoint"):
    rs = logs[s]; m = lambda k: np.mean([r[k] for r in rs])
    lines.append(f"| {s} | {m('majority_acc'):.3f} | {m('k1_random_f1'):.3f} | {m('k2_optimistic_f1'):.3f} | "
                 f"{m('k2_random_f1'):.3f} | {m('tfidf_lr_f1'):.3f} |")
open("results/summary_tables.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

os.makedirs("figures", exist_ok=True)

# Figure 2 of the report: published results vs. our valid results, full training data.
# Published values are those reported by Jiang et al. (2023); the paper's gzip numbers use
# top-2 (optimistic) scoring. Error bars are the 95% bootstrap intervals in results/stats.json.
stats = json.load(open("results/stats.json"))["full"]
PAPER_BERT = {"KinyarwandaNews": 0.838, "KirundiNews": 0.879, "DengueFilipino": 0.979,
              "SwahiliNews": 0.897, "R8": 0.982}
names = ["KinyarwandaNews", "KirundiNews", "DengueFilipino", "SwahiliNews", "R8"]
plt.rcParams.update({"font.family": "serif", "font.size": 11})
series = [  # label, values, (lo, hi) or None, colour, hatch
    ("Paper gzip (top-2 scoring)", [full[d]["paper"] for d in names], None, "#d9d9d9", "//"),
    ("Paper BERT", [PAPER_BERT[d] for d in names], None, "#7f7f7f", ""),
    ("Ours: gzip, k = 1 (valid)", [stats[d]["k1"][0] for d in names],
     [stats[d]["k1"][1:] for d in names], "#0072B2", ""),
    ("Ours: TF-IDF+LR", [stats[d]["tfidf_lr"][0] for d in names],
     [stats[d]["tfidf_lr"][1:] for d in names], "#E69F00", "")]
x = np.arange(len(names)); w = 0.2
fig, ax = plt.subplots(figsize=(9.3, 4.8))
for i, (lab, vals, ci, col, hatch) in enumerate(series):
    err = None if ci is None else np.array([[v - lo for v, (lo, hi) in zip(vals, ci)],
                                            [hi - v for v, (lo, hi) in zip(vals, ci)]])
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col, hatch=hatch, edgecolor="#555555",
           linewidth=0.6, yerr=err, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#333333"})
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.75, 1.0)
ax.set_ylabel("Test accuracy"); ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
fig.tight_layout(); fig.savefig("figures/fig2_published_vs_valid.png", dpi=200)
print("wrote figures/fig2_published_vs_valid.png")

# Figure 1 of the report: valid gzip + kNN accuracy as k varies
# (full training data, random tie-breaking, mean of five seeds; data from results/full.json).
ks = [1, 2, 3, 5, 7]
style = {"R8": ("#0072B2", "o"), "DengueFilipino": ("#D55E00", "s"),   # Okabe-Ito palette,
         "SwahiliNews": ("#009E73", "^"), "KinyarwandaNews": ("#CC79A7", "D"),  # colour-blind safe
         "KirundiNews": ("#E69F00", "v")}
plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(figsize=(9.3, 4.8))
ax.axvspan(1.75, 2.25, color="#e0e0e0", zorder=0)
ax.text(2, 0.936, "paper's k = 2", ha="center", fontsize=10, color="#555555")
ends = {}
for name, (col, mk) in style.items():
    ys = [full[name][f"k{k}_random"] for k in ks]
    ax.plot(ks, ys, color=col, marker=mk, lw=2, ms=6, zorder=3)
    ends[name] = ys[-1]
last = None  # end-of-line labels, nudged apart so they never overlap
for name, y in sorted(ends.items(), key=lambda kv: -kv[1]):
    ly = y if last is None else min(y, last - 0.0085); last = ly
    ax.text(7.2, ly, f"{name} {y:.3f}", va="center", fontsize=10, color=style[name][0])
ax.set_xticks(ks); ax.set_xlim(0.7, 9.6); ax.set_ylim(0.80, 0.94)
ax.set_xlabel("k (number of neighbours)"); ax.set_ylabel("Test accuracy")
ax.grid(axis="y", color="#e6e6e6"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig1_k_sensitivity.png", dpi=200)
print("wrote figures/fig1_k_sensitivity.png")
