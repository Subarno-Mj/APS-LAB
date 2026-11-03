"""Central configuration: paths, dataset names, label maps and thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Kaggle data
# 1) Ticket text for NLP triage (queue / priority / type)
TICKETS_KAGGLE = "tobiasbueck/multilingual-customer-support-tickets"
TICKETS_FILE = DATA_RAW / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"

# 2) ServiceNow incident event log for SLA-breach prediction (UCI #498 on Kaggle)
INCIDENTS_KAGGLE = "vipulshinde/incident-response-log"
INCIDENTS_FILE = DATA_RAW / "incident_event_log.csv"

SEED = 42

# ---------------------------------------------------------------- Triage
TRIAGE_TARGETS = ["queue", "priority", "type"]

# Auto-route a ticket for a target only if the calibrated confidence reaches
# the threshold that gives this accuracy on the validation set; otherwise the
# ticket goes to a human triage queue.
TRIAGE_TARGET_ACCURACY = {"queue": 0.85, "priority": 0.85, "type": 0.95}

# ---------------------------------------------------------------- SLA
# Text-model priority -> ITSM priority used by the incident log
PRIORITY_MAP = {"low": "4 - Low", "medium": "3 - Moderate", "high": "2 - High"}
ITSM_PRIORITIES = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]

# Risk bands on calibrated P(breach)
RISK_BANDS = [(0.0, 0.35, "LOW"), (0.35, 0.65, "MEDIUM"), (0.65, 1.01, "HIGH")]


def risk_band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "HIGH"
