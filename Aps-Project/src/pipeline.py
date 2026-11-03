"""End-to-end system: ticket text -> probabilistic triage -> SLA-breach risk -> action.

SLA risk for a new text ticket marginalises over the triage model's priority
uncertainty instead of trusting only its top guess:

    P(breach | text, ctx) = sum_k  P(priority=k | text) * P(breach | priority=k, ctx)

and reports the spread of the conditional risks as a measure of how much the
triage uncertainty matters for the SLA decision.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from . import config
from .sla_model import SLAModel
from .triage_model import TriageModel

ITSM_CONTEXT_FIELDS = ["impact", "urgency", "category", "subcategory", "u_symptom", "location",
                       "contact_type", "opened_by", "sys_created_by", "assignment_group"]


class SupportTriageSystem:
    def __init__(self, triage: TriageModel | None = None, sla: SLAModel | None = None):
        self.triage = triage or TriageModel.load()
        self.sla = sla or SLAModel.load()

    def _time_features(self, opened_at: datetime | None):
        t = opened_at or datetime.now()
        return {"hour": t.hour, "weekday": t.weekday(), "is_weekend": int(t.weekday() >= 5)}

    def process(self, subject: str, body: str, opened_at: datetime | None = None,
                context: dict | None = None) -> dict:
        context = {k: v for k, v in (context or {}).items() if v not in (None, "", "Unknown")}
        tri = self.triage.predict(subject, body)

        has_itsm = any(k in context for k in ITSM_CONTEXT_FIELDS)
        model_name = "open_full" if has_itsm else "open_triage"
        base = {**self._time_features(opened_at), **context}

        prio_dist = tri["priority"]["distribution"]  # low/medium/high -> prob
        rows, weights, labels = [], [], []
        for text_prio, p in prio_dist.items():
            itsm = config.PRIORITY_MAP.get(text_prio, "3 - Moderate")
            rows.append({**base, "priority": itsm})
            weights.append(p)
            labels.append((text_prio, itsm))
        cond = self.sla.predict(model_name, rows)
        w = np.array(weights) / np.sum(weights)
        risk = float(np.dot(w, cond))
        spread = float(np.sqrt(np.dot(w, (cond - risk) ** 2)))

        band = config.risk_band(risk)
        actions = []
        if tri["needs_human_review"]:
            actions.append("Send to human triage desk (low-confidence routing/priority).")
        else:
            actions.append(f"Auto-route to '{tri['queue']['label']}' queue.")
        if band == "HIGH":
            actions.append("SLA risk HIGH: escalate, assign senior agent, notify team lead.")
        elif band == "MEDIUM":
            actions.append("SLA risk MEDIUM: add to watch-list and re-score on each update.")
        else:
            actions.append("SLA risk LOW: standard handling.")
        if spread > 0.2:
            actions.append("Risk depends strongly on the uncertain priority: confirm priority manually.")

        return {
            "triage": tri,
            "sla": {
                "model": model_name,
                "breach_probability": risk,
                "risk_band": band,
                "uncertainty_from_priority": spread,
                "conditional_risk": [
                    {"text_priority": tp, "itsm_priority": ip, "p_priority": float(wi),
                     "p_breach_given_priority": float(c)}
                    for (tp, ip), wi, c in zip(labels, w, cond)
                ],
            },
            "actions": actions,
        }

    def rescore_live(self, record: dict) -> dict:
        """Re-score an open incident at an update (needs elapsed_h, reassignment_count, ...)."""
        p = float(self.sla.predict("live", [record])[0])
        return {"breach_probability": p, "risk_band": config.risk_band(p)}
