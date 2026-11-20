#!/usr/bin/env python3
"""Generate publication figures from released aggregate CSVs.
Run: python code/plot_figures.py --tables-dir results --baseline-csv results/BASELINE_Q57.csv --out figures
Requires the pinned numerical and plotting environment. No licensed records are read.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import platform
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.text import Text
import numpy as np

BLUE = "#245F7A"
ORANGE = "#E3AF7A"
GRAY = "#CFD8DE"
INK = "#202B33"
SUBTLE = "#536471"
GROUPS = ["neither", "representation_only", "meetings_only", "both"]
GROUP_LABELS = ["Neither", "Representation only", "Meetings only", "Both"]
T1 = "T1_Procedure_by_voice.csv"
T7 = "T7_Model_contrasts.csv"

CAPTIONS = {'Figure1_Procedure_profiles': "Figure 1. Procedural responses by employee-voice configuration. Weighted percentages among 6,548 eligible EU27 employees. Representation includes a union, works council or similar body; meetings provide regular opportunities to express views. The configurations refer to these two channels. Source: author’s calculations from EWCS 2024 V2; counts are in Table S5.", 'FigureS1_Adjusted_contrasts': "Figure S1. Adjusted differences in reporting a clear procedure across employee-voice configurations. Points represent percentage-point contrasts and whiskers show pointwise 95% confidence intervals. The both-versus-neither estimate is recovered from the existing contrast covariance (S6). The hollow diamond (*) identifies the covariance-derived both-versus-neither contrast. Source: author’s calculations from EWCS 2024 V2.", 'FigureS2_Paired_domains': "Figure S2. Exploratory comparison of employer-paid training and reported procedural clarity. All three estimates use the same 6,516 respondents, covariates and final weights. The paired difference incorporates within-person covariance. Training covers the preceding 12 months; procedural clarity concerns the current report. Source: author’s calculations from EWCS 2024 V2."}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def draw_forest(records, labels, stem, title, subtitle, xlabel, xlim, xticks,
                size, derived_index=None, footnote=None):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.33, right=.735, top=.90, bottom=.20)
    n = len(records)
    for i, r in enumerate(records):
        e, lo, hi = (float(r[k]) for k in ("estimate_pp", "lo95_pp", "hi95_pp"))
        assert lo < e < hi
        marker = "D" if i == derived_index else "o"
        artist = ax.errorbar(e, i, xerr=[[e-lo], [hi-e]], fmt=marker,
                            color=BLUE, ecolor=BLUE, elinewidth=1.8,
                            markersize=6.2, capsize=4, capthick=1.4,
                            markerfacecolor="white" if i == derived_index else BLUE,
                            markeredgewidth=1.4, zorder=3)
        artist.lines[0].set_gid(stem + "-point-" + str(i+1))
        artist.lines[2][0].set_gid(stem + "-interval-" + str(i+1))
        pos = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.065, i, f"{e:+.2f} [{lo:.2f}, {hi:.2f}]", transform=pos,
                ha="left", va="center", fontsize=10.6, color=INK)
    ax.text(1.065, 1.05, "Estimate [95% CI]", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, color=SUBTLE)
    ax.axvline(0, color="#7D8C95", linewidth=1, zorder=1)
    ax.set_yticks(range(n), labels)
    ax.set_ylim(n-.48, -.52)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color="#DBE2E6", linewidth=.7, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#ADB9C0")
    return fig


def save_figure(fig, stem, out):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    width, height = fig.bbox.width, fig.bbox.height
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > width+1 or b.y1 > height+1:
            outside.append(t.get_text())
    assert not outside, (stem, "Text outside image bounds", outside)
    result = {"size_inches": fig.get_size_inches().tolist(),
              "raster_dpi": 300, "text_bounds": "PASS", "outputs": {}}
    for ext in ("png", "svg"):
        path = out / (stem + "." + ext)
        metadata = {"Software": None} if ext == "png" else {
            "Creator": None, "Date": None,
            "Description": None,
        }
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text(path.read_text(encoding="utf-8").replace("matplotlib.axis_", "axis_"), encoding="utf-8")
        result["outputs"][path.name] = {"sha256": digest(path.read_bytes()),
                                        "bytes": path.stat().st_size}
    from PIL import Image
    path = out / (stem + ".tiff")
    with Image.open(out / (stem + ".png")) as image:
        image.save(path, format="TIFF", compression="tiff_deflate", dpi=(300, 300))
    result["outputs"][path.name] = {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = {T1: (args.tables_dir / T1).read_bytes(), T7: (args.tables_dir / T7).read_bytes()}
    t1, t7 = rows(source[T1]), rows(source[T7])
    baseline_bytes = args.baseline_csv.read_bytes()
    baseline = rows(baseline_bytes)
    assert len(t1) == 12 and len(t7) == 268 and len(baseline) == 9
    lookup = {(r["model"], r["contrast"]): r for r in t7}
    b = next(r for r in baseline if r["outcome"] == "procedure_yes"
             and r["contrast"] == "both - neither")
    assert b["provenance"] == "reconstructed_from_rounded_T7_contrasts_and_SEs"
    assert abs(float(b["estimate_pp"]) - float(lookup[("A_procedure_yes", "meetings_only - neither")]["estimate_pp"])
               - float(lookup[("A_procedure_yes", "both - meetings_only")]["estimate_pp"])) < 3e-8
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "ewcs-figures-2026-10-28",
        "svg.fonttype": "none", "axes.unicode_minus": True,
    })
    results = {}
    stem = "Figure1_Procedure_profiles"
    fig, ax = plt.subplots(figsize=(10.2, 3.85))
    fig.subplots_adjust(left=.235, right=.975, top=.97, bottom=.30)
    left = np.zeros(4)
    for level, label, color in [("1", "Reports clear procedure", BLUE),
                                 ("2", "Reports no clear procedure", ORANGE),
                                 ("3", "Does not know", GRAY)]:
        values = [float(next(r for r in t1 if r["group"] == g and r["outcome"] == level)["weighted_pct"])
                  for g in GROUPS]
        bars = ax.barh(range(4), values, left=left, label=label, color=color,
                       height=.65, linewidth=0)
        for i, (v, patch) in enumerate(zip(values, bars)):
            patch.set_gid("response-" + level + "-" + GROUPS[i])
            assert abs(patch.get_width()-v) < 1e-12
            ax.text(left[i]+v/2, i, f"{v:.1f}%", ha="center", va="center",
                    color="white" if level == "1" else INK, fontsize=11)
        left += values
    assert max(abs(left-100)) <= .000151
    ax.set_yticks(range(4), GROUP_LABELS)
    ax.set_ylim(3.6, -.6)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Weighted percentage of eligible employees", labelpad=12)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#ADB9C0")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .035),
               ncol=3, frameon=False, handlelength=1.35, columnspacing=1.7,
               fontsize=10.2)
    results[stem] = save_figure(fig, stem, args.out)
    results[stem]["values"] = t1

    selected = [lookup[("A_procedure_yes", c)] for c in (
        "representation_only - neither", "meetings_only - neither")]
    selected += [b, lookup[("A_procedure_yes", "both - meetings_only")],
                 lookup[("A_procedure_yes", "additive_interaction")]]
    stem = "FigureS1_Adjusted_contrasts"
    fig = draw_forest(selected,
        ["Representation only vs neither", "Meetings only vs neither",
         "Both vs neither*", "Both vs meetings only", "Additive interaction"],
        stem, "Reported procedural clarity: adjusted contrasts",
        "Differences in reporting a clear procedure (N = 6,548)",
        "Adjusted difference (percentage points)", (-32, 34), list(range(-30, 31, 10)),
        (10.5, 4.5), derived_index=2,
        footnote="* Secondary contrast recovered from model covariance.")
    results[stem] = save_figure(fig, stem, args.out)
    results[stem]["values"] = selected

    selected = [lookup[(m, "both - meetings_only")] for m in (
        "PAIRED_training", "PAIRED_procedure_yes", "PAIRED_training_minus_procedure")]
    stem = "FigureS2_Paired_domains"
    fig = draw_forest(selected,
        ["Employer-paid training", "Reports clear procedure",
         "Paired difference:\ntraining minus procedure"],
        stem, "Exploratory paired comparison",
        "Both channels versus meetings only on the same respondents (N = 6,516)",
        "Adjusted difference (percentage points)", (-15, 25), [-10, 0, 10, 20],
        (10.5, 3.7))
    results[stem] = save_figure(fig, stem, args.out)
    results[stem]["values"] = selected
    (args.out / "CAPTIONS.md").write_text("# Figure captions\n\n" +
         "\n\n".join(CAPTIONS.values()) + "\n", encoding="utf-8")
    receipt = {
        "boundary": "Aggregate-only figures. No raw records, empirical re-estimation, or new models.",
        "source_type": "Released aggregate CSVs",
        "source_table_sha256": {k: digest(v) for k, v in source.items()},
        "derived_baseline_sha256": digest(baseline_bytes),
        "plot_script_sha256": digest(Path(__file__).read_bytes()),
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                        "numpy": np.__version__},
        "figures": results,
        "visual_review": "Inspect the generated PNGs. Automated text bounds have passed.",
    }
    (args.out / "FIGURES_QA.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "FIGURES_WRITTEN", "figures": list(results),
                      "text_bounds": "PASS", "source_sha256": receipt["source_table_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
