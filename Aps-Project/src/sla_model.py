"""SLA-breach risk prediction on the ServiceNow incident event log.

Target: an incident BREACHES its SLA when its final `made_sla` flag is False.

Three calibrated models are trained:
  * open_full    – risk at the moment a ticket is opened, using all ITSM fields
                   known at that time (priority, impact, urgency, category,
                   sub-category, symptom, location, assignment group, ...)
                   plus workload context (open backlog, arrivals last 24 h).
  * open_triage  – the same moment, but only with what the NLP triage step can
                   supply (priority) + time + workload. Used end-to-end for new
                   text tickets.
  * live         – re-scores a ticket at every update while it is still within
                   SLA (elapsed time, reassignments, reopens, updates so far).

Leakage control: columns that only get their final value at closure
(closed_code, resolved_by, resolved_at, closed_at, knowledge, problem_id, rfc,
vendor, caused_by, u_priority_confirmation, assigned_to) are never used, and the
train/val/test split is chronological on opened_at.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .metrics import expected_calibration_error

DATE_FMT = "%d/%m/%Y %H:%M"
OPEN_STATES = {"New", "Active", "Awaiting User Info", "Awaiting Problem",
               "Awaiting Vendor", "Awaiting Evidence"}

CAT_FULL = ["priority", "impact", "urgency", "category", "subcategory", "u_symptom",
            "location", "contact_type", "opened_by", "sys_created_by", "assignment_group"]
NUM_OPEN = ["hour", "weekday", "is_weekend", "registration_lag_min", "backlog_open",
            "backlog_group", "arrivals_24h", "caller_prior_incidents"]
CAT_TRIAGE = ["priority"]
NUM_TRIAGE = ["hour", "weekday", "is_weekend", "backlog_open", "arrivals_24h"]
CAT_LIVE = CAT_FULL + ["incident_state", "current_group"]
NUM_LIVE = NUM_OPEN + ["elapsed_h", "reassignment_count", "reopen_count", "sys_mod_count"]

FEATURE_SETS = {
    "open_full": (CAT_FULL, NUM_OPEN),
    "open_triage": (CAT_TRIAGE, NUM_TRIAGE),
    "live": (CAT_LIVE, NUM_LIVE),
}


# ------------------------------------------------------------------ features
def _dt(s):
    return pd.to_datetime(s.replace("?", np.nan), format=DATE_FMT, errors="coerce")


def _secs(s):
    """Datetime series -> int64 epoch seconds (independent of pandas' time unit)."""
    return (s - pd.Timestamp("1970-01-01")).dt.total_seconds().fillna(4e9).astype("int64").values


def _open_count(t_query, starts, ends):
    """# intervals with start < t and end > t, for each t (vectorised)."""
    s, e, q = np.sort(_secs(starts)), np.sort(_secs(ends)), _secs(t_query)
    return np.clip(np.searchsorted(s, q, side="left") - np.searchsorted(e, q, side="right"), 0, None)


def build_tables(path=config.INCIDENTS_FILE):
    ev = pd.read_csv(path, na_values=[], keep_default_na=False)
    for c in ["opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at"]:
        ev[c] = _dt(ev[c])
    ev = ev.sort_values(["number", "sys_mod_count", "sys_updated_at"]).reset_index(drop=True)

    first = ev.groupby("number").head(1).set_index("number")
    last = ev.groupby("number").tail(1).set_index("number")

    inc = first[CAT_FULL + ["opened_at", "sys_created_at", "caller_id"]].copy()
    inc = inc.replace("?", "Unknown")
    inc["breach"] = (~last["made_sla"].astype(str).str.lower().eq("true")).astype(int)
    inc["end_at"] = last["resolved_at"].fillna(last["closed_at"])
    inc["resolution_h"] = (inc["end_at"] - inc["opened_at"]).dt.total_seconds() / 3600

    t = inc["opened_at"]
    inc["hour"] = t.dt.hour
    inc["weekday"] = t.dt.weekday
    inc["is_weekend"] = (inc["weekday"] >= 5).astype(int)
    inc["registration_lag_min"] = ((inc["sys_created_at"] - t).dt.total_seconds() / 60).clip(lower=0)

    ends = inc["end_at"]
    inc["backlog_open"] = _open_count(t, t, ends)
    inc["backlog_group"] = 0
    for g, idx in inc.groupby("assignment_group").groups.items():
        sub = inc.loc[idx]
        inc.loc[idx, "backlog_group"] = _open_count(sub["opened_at"], sub["opened_at"], ends.loc[idx])
    q = _secs(t)
    ts = np.sort(q)
    inc["arrivals_24h"] = np.searchsorted(ts, q, "left") - np.searchsorted(ts, q - 86_400, "left")
    inc = inc.sort_values("opened_at")
    inc["caller_prior_incidents"] = inc.groupby("caller_id").cumcount()

    # live (event-level) table: still open AND currently within SLA
    live = ev[ev["incident_state"].isin(OPEN_STATES) & ev["made_sla"].astype(str).str.lower().eq("true")].copy()
    live = live[["number", "incident_state", "assignment_group", "sys_updated_at",
                 "reassignment_count", "reopen_count", "sys_mod_count"]]
    live = live.rename(columns={"assignment_group": "current_group"}).replace("?", "Unknown")
    live = live.join(inc.drop(columns=["end_at", "resolution_h"]), on="number")
    live["elapsed_h"] = ((live["sys_updated_at"] - live["opened_at"]).dt.total_seconds() / 3600).clip(lower=0)
    live = live.dropna(subset=["elapsed_h"])
    return inc.reset_index(), live.reset_index(drop=True)


def chrono_split(df, time_col="opened_at", val=0.10, test=0.20):
    q_val, q_test = df[time_col].quantile([1 - val - test, 1 - test])
    tr = df[time_col] < q_val
    va = (df[time_col] >= q_val) & (df[time_col] < q_test)
    te = df[time_col] >= q_test
    return tr.values, va.values, te.values


# ------------------------------------------------------------------ models
class CategoryCapper:
    """Map rare/unseen categories to 'Other' so every column has <= max_levels levels."""

    def __init__(self, min_count=20, max_levels=200):
        self.min_count, self.max_levels, self.levels = min_count, max_levels, {}

    def fit(self, df, cols):
        for c in cols:
            vc = df[c].astype(str).value_counts()
            self.levels[c] = list(vc[vc >= self.min_count].index[: self.max_levels])
        return self

    def transform(self, df):
        out = df.copy()
        for c, lv in self.levels.items():
            s = out[c].astype(str)
            out[c] = s.where(s.isin(lv), "Other")
        return out


def _make_hgb(cat_cols, num_cols):
    from sklearn.preprocessing import OrdinalEncoder
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    mask = [True] * len(cat_cols) + [False] * len(num_cols)
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         categorical_features=mask, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=30,
                                         random_state=config.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def _make_lr(cat_cols, num_cols):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=3000))])


def _binary_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(np.mean(y)),
    }


class SLAModel:
    def __init__(self):
        self.models, self.calibrators, self.cappers, self.kinds = {}, {}, {}, {}
        self.defaults, self.report, self.test_preds = {}, {}, {}

    def _fit_one(self, name, df, verbose=True):
        cat_cols, num_cols = FEATURE_SETS[name]
        tr, va, te = chrono_split(df)
        capper = CategoryCapper().fit(df[tr], cat_cols)
        X = capper.transform(df[cat_cols + num_cols])
        X[num_cols] = X[num_cols].astype(float)
        y = df["breach"].values

        # for the live table, keep one event in five during training for speed
        res = {}
        best = None
        for kind, maker in (("logistic_regression", _make_lr), ("gradient_boosting", _make_hgb)):
            m = maker(cat_cols, num_cols).fit(X[tr], y[tr])
            p_va_raw, p_te_raw = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p_va_raw, y[va])
            p_te = iso.predict(p_te_raw)
            mt = _binary_metrics(y[te], p_te)
            mt["ece_uncalibrated"] = expected_calibration_error(y[te], p_te_raw)
            mt["val_roc_auc"] = float(roc_auc_score(y[va], p_va_raw))
            res[kind] = mt
            if best is None or mt["val_roc_auc"] > res[best[0]]["val_roc_auc"]:
                best = (kind, m, iso, p_te)
        kind, m, iso, p_te = best
        self.models[name], self.calibrators[name], self.cappers[name], self.kinds[name] = m, iso, capper, kind
        self.test_preds[name] = {"y": y[te], "p": p_te,
                                 "priority": df.loc[te, "priority"].values}
        # priority-only baseline (what a static SLA rule would know)
        pr_rate = df[tr].groupby("priority")["breach"].mean()
        p_base = df.loc[te, "priority"].map(pr_rate).fillna(df.loc[tr, "breach"].mean()).values
        res["priority_rate_baseline"] = _binary_metrics(y[te], p_base)
        res["chosen_model"] = kind
        res["n_train"], res["n_val"], res["n_test"] = int(tr.sum()), int(va.sum()), int(te.sum())
        bands = pd.cut(p_te, [b[0] for b in config.RISK_BANDS] + [1.01],
                       labels=[b[2] for b in config.RISK_BANDS], right=False)
        res["risk_bands"] = (pd.DataFrame({"band": bands, "breach": y[te]})
                             .groupby("band", observed=False)["breach"].agg(["count", "mean"])
                             .rename(columns={"mean": "observed_breach_rate"})
                             .reset_index().to_dict("records"))
        self.report[name] = res
        if verbose:
            b = res[kind]
            print(f"[sla:{name:11s}] best={kind:19s} AUC={b['roc_auc']:.3f} PR-AUC={b['pr_auc']:.3f} "
                  f"Brier={b['brier']:.3f} ECE {b['ece_uncalibrated']:.3f}->{b['ece']:.3f} "
                  f"| priority-only AUC={res['priority_rate_baseline']['roc_auc']:.3f}")
        return X[te], y[te]

    def fit_evaluate(self, inc: pd.DataFrame, live: pd.DataFrame, verbose=True) -> dict:
        Xte, yte = self._fit_one("open_full", inc, verbose)
        self._fit_one("open_triage", inc, verbose)
        live_s = live.sample(frac=0.35, random_state=config.SEED) if len(live) > 60_000 else live
        self._fit_one("live", live_s, verbose)

        # permutation importance of the open-time model (on AUC)
        m = self.models["open_full"]
        imp = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=5,
                                     random_state=config.SEED, n_jobs=1)
        cols = FEATURE_SETS["open_full"][0] + FEATURE_SETS["open_full"][1]
        self.report["open_full"]["permutation_importance"] = dict(
            sorted(zip(cols, imp.importances_mean.round(4).tolist()), key=lambda z: -z[1]))

        num = NUM_OPEN + ["elapsed_h", "reassignment_count", "reopen_count", "sys_mod_count"]
        self.defaults = {c: float((inc[c] if c in inc else live[c]).median()) for c in num}
        self.category_levels = {n: c.levels for n, c in self.cappers.items()}
        # breach rate by priority, Beta(1,1)-smoothed – used for explanations
        g = inc.groupby("priority")["breach"].agg(["sum", "count"])
        self.priority_breach_rate = ((g["sum"] + 1) / (g["count"] + 2)).to_dict()
        self.report["breach_rate_by_priority"] = self.priority_breach_rate
        self.report["overall_breach_rate"] = float(inc["breach"].mean())
        return self.report

    # ---------------------------------------------------------------- predict
    def predict(self, name: str, records: list[dict] | pd.DataFrame) -> np.ndarray:
        cat_cols, num_cols = FEATURE_SETS[name]
        df = pd.DataFrame(records).copy()
        for c in cat_cols:
            if c not in df:
                df[c] = "Unknown"
        for c in num_cols:
            if c not in df:
                df[c] = self.defaults.get(c, 0.0)
        X = self.cappers[name].transform(df[cat_cols + num_cols])
        X[num_cols] = X[num_cols].astype(float)
        raw = self.models[name].predict_proba(X)[:, 1]
        return self.calibrators[name].predict(raw)

    def save(self, path=config.MODELS / "sla_model.joblib"):
        self.test_preds = {}
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=config.MODELS / "sla_model.joblib") -> "SLAModel":
        return joblib.load(path)
