"""Figures and markdown report for the trained system."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from . import config
from .metrics import reliability_curve

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=150)
    plt.close(fig)


def triage_figures(triage, raw_preds: dict):
    """raw_preds[target] = {'y', 'p_cal', 'p_nb_raw', 'p_nb_cal'} on the test split."""
    # 1. coverage vs accuracy (selective prediction)
    fig, ax = plt.subplots(figsize=(6, 4))
    for tgt, col in zip(config.TRIAGE_TARGETS, (BLUE, ORANGE, AQUA)):
        curve = triage.targets[tgt].metrics["coverage_curve"]
        ax.plot([c["coverage"] for c in curve], [c["accuracy"] for c in curve], color=col, lw=2, label=tgt)
        cov = triage.targets[tgt].metrics["selective"]["test_coverage_auto_routed"]
        acc = triage.targets[tgt].metrics["selective"]["test_accuracy_auto_routed"] or 0
        ax.plot([cov], [acc], "o", color=col, ms=8, mec="#fcfcfb", mew=2)
    ax.set_xlabel("Share of tickets auto-routed (coverage)")
    ax.set_ylabel("Accuracy on auto-routed tickets")
    ax.set_title("Selective triage: confidence threshold trades coverage for accuracy")
    ax.legend(loc="lower left")
    _save(fig, "triage_coverage_accuracy.png")

    # 2. reliability diagram: NB raw vs NB calibrated vs LR calibrated (queue)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, tgt in zip(axes, config.TRIAGE_TARGETS):
        d = raw_preds[tgt]
        y = np.array(d["y"])
        for key, col, lab in (("p_nb_raw", ORANGE, "Naive Bayes (raw)"),
                              ("p_nb_cal", YELLOW, "Naive Bayes + temperature"),
                              ("p_cal", BLUE, f"Chosen model ({triage.targets[tgt].model_kind.replace('_', ' ')})")):
            p = np.array(d[key])
            conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
            mp, of, _ = reliability_curve(corr, conf, 10)
            ax.plot(mp, of, "-o", color=col, lw=2, ms=5, label=lab)
        ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
        ax.set_title(f"{tgt}: confidence vs. observed accuracy")
        ax.set_xlabel("Predicted confidence")
    axes[0].set_ylabel("Observed accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    _save(fig, "triage_reliability.png")

    # 3. queue confusion matrix (row-normalised)
    tm = triage.targets["queue"]
    y = np.array(raw_preds["queue"]["y"])
    p = np.array(raw_preds["queue"]["p_cal"]).argmax(1)
    cm = confusion_matrix(y, p, normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(tm.classes)), tm.classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tm.classes)), tm.classes, fontsize=8)
    for i in range(len(cm)):
        ax.text(i, i, f"{cm[i, i]:.2f}", ha="center", va="center",
                color="white" if cm[i, i] > 0.5 else INK, fontsize=8)
    ax.set_xlabel("Predicted queue")
    ax.set_ylabel("True queue")
    ax.set_title("Queue routing: row-normalised confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "triage_queue_confusion.png")


def sla_figures(sla):
    # ROC for the three models
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, col in zip(("open_full", "open_triage", "live"), (BLUE, ORANGE, AQUA)):
        d = sla.test_preds[name]
        fpr, tpr, _ = roc_curve(d["y"], d["p"])
        auc = sla.report[name][sla.kinds[name]]["roc_auc"]
        axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{name}  AUC={auc:.3f}")
        mp, of, cnt = reliability_curve(d["y"], d["p"], 10)
        axes[1].plot(mp, of, "-o", color=col, lw=2, ms=5, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("SLA-breach ROC (chronological test set)")
    axes[0].legend(loc="lower right")
    axes[1].plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    axes[1].set_xlabel("Predicted P(breach)")
    axes[1].set_ylabel("Observed breach rate")
    axes[1].set_title("Calibration after isotonic regression")
    axes[1].legend(loc="upper left")
    _save(fig, "sla_roc_calibration.png")

    imp = sla.report["open_full"]["permutation_importance"]
    items = [(k, v) for k, v in imp.items()][:12][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, height=0.6)
    ax.set_xlabel("Drop in test ROC-AUC when feature is shuffled")
    ax.set_title("What drives SLA-breach risk at ticket open")
    _save(fig, "sla_feature_importance.png")


def write_markdown(report: dict, triage, sla):
    L = ["# Results — Probabilistic NLP Support Triage & SLA Risk Prediction", ""]
    L += ["All numbers are on held-out test data. Triage: stratified 70/10/20 split. "
          "SLA: chronological 70/10/20 split on `opened_at` (train on the past, test on the future).", ""]
    L += ["## 1. Probabilistic triage (ticket text → queue / priority / type)", "",
          "| Target | Classes | Majority baseline | Naive Bayes acc | LogReg acc | Chosen | Macro-F1 | ECE raw → calibrated | Auto-routed | Accuracy when auto-routed | Accuracy sent to human |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in config.TRIAGE_TARGETS:
        r = report["triage"]["targets"][t]
        c = r["candidates"]
        ch = c[r["chosen_model"]]
        s = r["selective"]
        L.append(f"| {t} | {len(triage.targets[t].classes)} | {r['majority_baseline']:.3f} | "
                 f"{c['naive_bayes']['accuracy']:.3f} | {c['logistic_regression']['accuracy']:.3f} | "
                 f"{r['chosen_model'].replace('_', ' ')} | {ch['macro_f1']:.3f} | "
                 f"{ch['ece_raw']:.3f} → {ch['ece_calibrated']:.3f} | {s['test_coverage_auto_routed']:.1%} | "
                 f"{(s['test_accuracy_auto_routed'] or 0):.3f} | {(s['test_accuracy_sent_to_human'] or 0):.3f} |")
    nb = report["triage"]["targets"]["queue"]["candidates"]["naive_bayes"]
    L += ["", f"Naive Bayes is strongly over-confident (queue ECE {nb['ece_raw']:.3f}); temperature scaling "
          f"(T = {nb['temperature']:.2f}) fixes it (ECE {nb['ece_calibrated']:.3f}). Logistic regression is "
          "close to calibrated out of the box.", "",
          "![coverage](figures/triage_coverage_accuracy.png)", "",
          "![reliability](figures/triage_reliability.png)", "",
          "![confusion](figures/triage_queue_confusion.png)", ""]
    L += ["### Most indicative terms per queue", ""]
    for cls, terms in triage.top_terms("queue", 8).items():
        L.append(f"- **{cls}**: {', '.join(terms)}")
    L += ["", "## 2. SLA-breach risk (ServiceNow incident log)", "",
          f"Overall breach rate: {report['sla']['overall_breach_rate']:.1%}. "
          "Breach rate by priority (Beta(1,1)-smoothed): " +
          ", ".join(f"{k}: {v:.1%}" for k, v in sorted(report['sla']['breach_rate_by_priority'].items())), "",
          "| Model | When it is used | Chosen | ROC-AUC | PR-AUC | Brier | ECE before → after isotonic | Priority-only rule AUC | Test base rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    use = {"open_full": "ticket opened, ITSM fields known",
           "open_triage": "ticket opened, only text-triage priority + time + workload",
           "live": "every update while still within SLA"}
    for n in ("open_full", "open_triage", "live"):
        r = report["sla"][n]
        b = r[r["chosen_model"]]
        L.append(f"| {n} | {use[n]} | {r['chosen_model'].replace('_', ' ')} | {b['roc_auc']:.3f} | {b['pr_auc']:.3f} | "
                 f"{b['brier']:.3f} | {b['ece_uncalibrated']:.3f} → {b['ece']:.3f} | "
                 f"{r['priority_rate_baseline']['roc_auc']:.3f} | {b['base_rate']:.1%} |")
    L += ["", "Risk bands on the open-time test set (does a HIGH band really breach more?):", "",
          "| Band | Tickets | Observed breach rate |", "|---|---|---|"]
    for row in report["sla"]["open_full"]["risk_bands"]:
        L.append(f"| {row['band']} | {row['count']} | {row['observed_breach_rate'] if row['observed_breach_rate'] == row['observed_breach_rate'] else 0:.1%} |")
    L += ["", "![roc](figures/sla_roc_calibration.png)", "", "![imp](figures/sla_feature_importance.png)", "",
          "## 3. Honest limitations", "",
          "- The ticket-text dataset is synthetic (LLM-generated, stated on Kaggle); its labels are noisy, "
          "which caps queue accuracy near 60%. That is why the system abstains and hands low-confidence tickets to a human.",
          "- The two datasets are different organisations. They are linked only through priority "
          "(low→4-Low, medium→3-Moderate, high→2-High), an explicit assumption in `src/config.py`.",
          "- With priority alone, SLA risk is only weakly predictable (open_triage AUC above). The useful signal "
          "comes from routing context: assignment group, symptom, location. Supply them when known.",
          "- The incident log's categories and groups are anonymised IDs (e.g. 'Group 56').",
          "- The commonly used Kaggle 'Customer Support Ticket Dataset' (suraj520) was evaluated and rejected: "
          "text models scored at chance (Ticket Type 19.4% vs 20.7% majority) and 49% of its resolution timestamps "
          "precede the first response, i.e. the labels are random.", ""]
    (config.REPORTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    with open(config.REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
