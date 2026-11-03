"""Probabilistic NLP triage: P(queue | text), P(priority | text), P(type | text).

Pipeline
  text -> cleaning -> TF-IDF (word 1-2 grams + char 3-5 grams)
       -> two probabilistic classifiers compared per target:
            * Multinomial Naive Bayes  (generative, P(words | class) P(class))
            * Multinomial Logistic Regression (discriminative, softmax)
       -> temperature scaling on a validation split (calibrated probabilities)
       -> selective prediction: auto-route only when confidence >= threshold
          chosen on validation to reach a target accuracy, else human review.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from . import config
from .metrics import expected_calibration_error, multiclass_brier, normalized_entropy
from .text_preprocess import ticket_text


# ------------------------------------------------------------------ data
def load_tickets(path=config.TICKETS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["body", "queue", "priority", "type"]).reset_index(drop=True)
    df["text"] = [ticket_text(s, b) for s, b in zip(df["subject"], df["body"])]
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    return df


# ------------------------------------------------------------------ features
class TextFeaturizer:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150_000,
                                    sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    max_features=150_000, sublinear_tf=True)

    def fit(self, texts):
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


# ------------------------------------------------------------------ calibration
def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def apply_temperature(proba, T):
    return _softmax(np.log(np.clip(proba, 1e-12, 1)) / T)


def fit_temperature(proba, y_idx) -> float:
    """Single scalar T minimising validation negative log-likelihood."""
    logp = np.log(np.clip(proba, 1e-12, 1))

    def nll(log_t):
        p = _softmax(logp / np.exp(log_t))
        return -np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1)).mean()

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def pick_threshold(conf, correct, target_acc) -> float:
    """Lowest confidence threshold whose accepted set reaches target accuracy."""
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.where(cum_acc >= target_acc)[0]
    ok = ok[ok >= 30]  # need a minimally sized accepted set
    if len(ok) == 0:
        return 1.01
    return float(conf[order][ok.max()])


# ------------------------------------------------------------------ model
@dataclass
class TargetModel:
    name: str
    classes: list
    model: object
    model_kind: str
    temperature: float
    threshold: float
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, X):
        return apply_temperature(self.model.predict_proba(X), self.temperature)


class TriageModel:
    def __init__(self):
        self.featurizer = TextFeaturizer()
        self.targets: dict[str, TargetModel] = {}

    # ---------------------------------------------------------------- train
    def fit_evaluate(self, df: pd.DataFrame, verbose=True) -> dict:
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.20, random_state=config.SEED,
                                  stratify=df["queue"])
        tr, va = train_test_split(tr, test_size=0.125, random_state=config.SEED,
                                  stratify=df["queue"].iloc[tr])  # 70/10/20
        self.featurizer.fit(df["text"].iloc[tr])
        Xtr, Xva, Xte = (self.featurizer.transform(df["text"].iloc[i]) for i in (tr, va, te))

        report = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "targets": {}}
        self._test_idx = te
        for tgt in config.TRIAGE_TARGETS:
            y = df[tgt].astype(str).values
            classes = sorted(np.unique(y))
            c2i = {c: i for i, c in enumerate(classes)}
            ytr, yva, yte = (np.array([c2i[v] for v in y[i]]) for i in (tr, va, te))

            candidates = {
                "naive_bayes": MultinomialNB(alpha=0.05),
                "logistic_regression": LogisticRegression(C=8.0, max_iter=3000),
            }
            cand_report, best, cand_raw = {}, None, {}
            for kind, clf in candidates.items():
                clf.fit(Xtr, ytr)
                p_va_raw = clf.predict_proba(Xva)
                T = fit_temperature(p_va_raw, yva)
                p_va = apply_temperature(p_va_raw, T)
                p_te_raw = clf.predict_proba(Xte)
                p_te = apply_temperature(p_te_raw, T)
                pred = p_te.argmax(1)
                m = {
                    "temperature": T,
                    "accuracy": accuracy_score(yte, pred),
                    "macro_f1": f1_score(yte, pred, average="macro"),
                    "log_loss_raw": log_loss(yte, p_te_raw, labels=range(len(classes))),
                    "log_loss_calibrated": log_loss(yte, p_te, labels=range(len(classes))),
                    "ece_raw": expected_calibration_error(yte, p_te_raw),
                    "ece_calibrated": expected_calibration_error(yte, p_te),
                    "brier_calibrated": multiclass_brier(yte, p_te),
                    "val_log_loss": log_loss(yva, p_va, labels=range(len(classes))),
                }
                cand_report[kind] = m
                cand_raw[kind] = (p_te_raw, T)
                if best is None or m["val_log_loss"] < cand_report[best[0]]["val_log_loss"]:
                    best = (kind, clf, T, p_va, p_te)

            kind, clf, T, p_va, p_te = best
            thr = pick_threshold(p_va.max(1), (p_va.argmax(1) == yva).astype(float),
                                 config.TRIAGE_TARGET_ACCURACY[tgt])
            conf_te = p_te.max(1)
            auto = conf_te >= thr
            sel = {
                "threshold": thr,
                "target_accuracy": config.TRIAGE_TARGET_ACCURACY[tgt],
                "test_coverage_auto_routed": float(auto.mean()),
                "test_accuracy_auto_routed": float((p_te.argmax(1)[auto] == yte[auto]).mean()) if auto.any() else None,
                "test_accuracy_sent_to_human": float((p_te.argmax(1)[~auto] == yte[~auto]).mean()) if (~auto).any() else None,
                "mean_entropy_auto": float(normalized_entropy(p_te[auto]).mean()) if auto.any() else None,
                "mean_entropy_human": float(normalized_entropy(p_te[~auto]).mean()) if (~auto).any() else None,
            }
            # coverage-accuracy curve for the report
            order = np.argsort(-conf_te)
            corr = (p_te.argmax(1) == yte)[order]
            cov = np.arange(1, len(order) + 1) / len(order)
            acc_curve = np.cumsum(corr) / np.arange(1, len(order) + 1)
            pts = np.linspace(0.05, 1.0, 20)
            curve = [{"coverage": float(c), "accuracy": float(acc_curve[int(c * len(order)) - 1])} for c in pts]

            self.targets[tgt] = TargetModel(tgt, classes, clf, kind, T, thr,
                                            {"chosen_model": kind, "candidates": cand_report,
                                             "selective": sel, "coverage_curve": curve,
                                             "majority_baseline": float(pd.Series(yte).value_counts(normalize=True).max())})
            # keep test predictions for plots
            nb_raw, nb_T = cand_raw["naive_bayes"]
            self.targets[tgt].metrics["_test"] = {"y": yte, "p_cal": p_te, "p_nb_raw": nb_raw,
                                                  "p_nb_cal": apply_temperature(nb_raw, nb_T)}
            report["targets"][tgt] = {k: v for k, v in self.targets[tgt].metrics.items() if k != "_test"}
            if verbose:
                cm = cand_report[kind]
                print(f"[triage:{tgt:8s}] best={kind:19s} acc={cm['accuracy']:.3f} "
                      f"macroF1={cm['macro_f1']:.3f} ECE {cm['ece_raw']:.3f}->{cm['ece_calibrated']:.3f} "
                      f"| auto-route {sel['test_coverage_auto_routed']:.1%} @ acc "
                      f"{sel['test_accuracy_auto_routed'] or 0:.3f}")
        return report

    # ---------------------------------------------------------------- predict
    def predict(self, subject: str, body: str) -> dict:
        X = self.featurizer.transform([ticket_text(subject, body)])
        out = {}
        for tgt, tm in self.targets.items():
            p = tm.predict_proba(X)[0]
            k = int(p.argmax())
            out[tgt] = {
                "label": tm.classes[k],
                "confidence": float(p[k]),
                "entropy": float(normalized_entropy(p[None])[0]),
                "auto_route": bool(p[k] >= tm.threshold),
                "threshold": tm.threshold,
                "distribution": {c: float(v) for c, v in sorted(zip(tm.classes, p), key=lambda z: -z[1])},
            }
        out["needs_human_review"] = not (out["queue"]["auto_route"] and out["priority"]["auto_route"])
        return out

    def top_terms(self, target: str, n: int = 12) -> dict:
        """Most indicative word n-grams per class (LR coefficients or NB log-prob ratios)."""
        tm = self.targets[target]
        vocab = np.array(self.featurizer.word.get_feature_names_out())
        nw = len(vocab)
        if tm.model_kind == "logistic_regression":
            W = tm.model.coef_[:, :nw]
        else:
            lp = tm.model.feature_log_prob_[:, :nw]
            W = lp - np.log(np.exp(lp).mean(0, keepdims=True))
        return {c: vocab[np.argsort(-W[i])[:n]].tolist() for i, c in enumerate(tm.classes)}

    def save(self, path=config.MODELS / "triage_model.joblib"):
        for tm in self.targets.values():
            tm.metrics.pop("_test", None)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=config.MODELS / "triage_model.joblib") -> "TriageModel":
        return joblib.load(path)
