# Probabilistic NLP-based Intelligent Support Triage & SLA Risk Prediction

APS project · Subarno Majumder · M.E. Big Data Analytics, MSIS Manipal

A support ticket arrives as free text. The system:

1. **Triages it probabilistically.** It predicts calibrated distributions P(queue | text),
   P(priority | text) and P(type | text) with Naive Bayes and multinomial logistic regression
   over TF-IDF word and character n-grams, then applies temperature scaling.
2. **Knows when it doesn't know.** A ticket is auto-routed only when the calibrated confidence
   clears a threshold tuned on validation data to hit a target accuracy. Every other ticket goes
   to a human triage desk, and predictive entropy is reported for each one.
3. **Predicts SLA-breach risk.** It gives a calibrated P(breach) using gradient boosting or
   logistic regression with isotonic calibration, trained on a real ServiceNow incident log.
   The risk is **marginalised over the triage uncertainty**:
   `P(breach | text) = Σ_k P(priority=k | text) · P(breach | priority=k, context)`.
4. **Re-scores open tickets live** at each update, using elapsed time, reassignments, reopens and
   update count.
5. **Recommends an action**: auto-route or human review, and escalate, watch or standard handling.

```
 ticket text ──► clean ──► TF-IDF (word 1-2g + char 3-5g)
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
          P(queue)       P(priority)        P(type)     NB vs LogReg + temperature scaling
              │               │
     confidence ≥ τ ?         │  (marginalise over priority)
      yes: auto-route         ▼
      no : human desk   P(breach | priority, time, workload, ITSM context)   isotonic-calibrated
                              │
                              ▼
                 risk band LOW / MEDIUM / HIGH ──► actions;  live re-scoring on every update
```

## Data (Kaggle)

| Use | Kaggle dataset | Size | Labels |
|---|---|---|---|
| NLP triage | [Customer IT Support – Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) (Tobias Bueck, CC BY 4.0) | 28,587 tickets, English + German | queue (10), priority (3), type (4) |
| SLA risk | [Incident Response Log](https://www.kaggle.com/datasets/vipulshinde/incident-response-log) (ServiceNow event log, UCI #498) | 119,998 events / 20,769 incidents | `made_sla` (final value = target) |

**Why not the popular "Customer Support Ticket Dataset" (suraj520)?** I tested it first. Text
classifiers score at chance there (Ticket Type 19.4% vs a 20.7% majority baseline), and 49% of
its resolution timestamps come before the first response. Its labels are random, so nothing
real can be learned from it.

## How to run (Windows, from this folder)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   :: shipped models were saved with scikit-learn 1.7.2; on another version run train.py

python data_download.py      :: skips download if data\raw already has both CSVs
python train.py              :: ~5-10 min CPU; writes models\ and reports\
streamlit run app.py         :: web UI
python predict.py --subject "Server down" --body "Our main server is not responding since morning"
```

## Project layout

```
APS Project/
├── app.py               Streamlit UI (triage, live SLA re-scoring, model report)
├── train.py             trains + evaluates everything, writes reports/RESULTS.md
├── predict.py           CLI demo
├── data_download.py     fetches both Kaggle datasets (kagglehub)
├── requirements.txt
├── src/
│   ├── config.py        paths, thresholds, priority mapping, risk bands
│   ├── text_preprocess.py
│   ├── triage_model.py  probabilistic NLP triage + calibration + selective routing
│   ├── sla_model.py     feature engineering (leak-free), SLA models, isotonic calibration
│   ├── pipeline.py      end-to-end system with risk marginalisation
│   ├── metrics.py       ECE, Brier, entropy, reliability curves
│   └── reports.py       figures + markdown report
├── data/raw/            Kaggle CSVs
├── models/              trained .joblib files
└── reports/             RESULTS.md, metrics.json, figures/
```

## Results

See **[reports/RESULTS.md](reports/RESULTS.md)** for every number and figure. Headline results
are below (held-out test set).

| Component | Metric | Result | Baseline |
|---|---|---|---|
| Queue routing (10 classes) | accuracy / macro-F1 | 0.589 / 0.545 | majority 0.292 |
| Queue, selective auto-routing | accuracy on the 29.6% auto-routed | **0.838** | — |
| Priority (3 classes) | accuracy / macro-F1 | 0.635 / 0.617 | majority 0.401 |
| Ticket type (4 classes) | accuracy, auto-routed share | 0.868; **0.962** on the 65.9% auto-routed | majority 0.405 |
| Calibration | Naive Bayes queue ECE, raw → temperature-scaled | 0.363 → 0.024 | — |
| SLA risk at open (full ITSM context) | ROC-AUC / ECE | **0.781** / 0.060 | priority-only rule 0.547 |
| SLA risk, live re-scoring | ROC-AUC | **0.794** | priority-only rule 0.531 |
| Risk bands (open-time test set) | observed breach rate LOW / MEDIUM / HIGH | 17% / 48% / 82% | overall 22% |

## Design decisions worth defending in a viva

- **Calibration matters as much as accuracy.** The routing threshold and the SLA marginalisation
  both read the probabilities as probabilities, so the project reports ECE, Brier score and
  log-loss alongside accuracy and AUC. Naive Bayes shows why: it is badly overconfident until
  temperature scaling fixes it.
- **Abstention instead of forcing a label.** The queue labels are noisy, so the system routes
  the confident share automatically at high accuracy and sends the rest to people.
- **No leakage.** Incident-log fields that are only final at closure (`closed_code`, `resolved_by`,
  `resolved_at`, `knowledge`, `problem_id`, `rfc`, `vendor`, `caused_by`, `u_priority_confirmation`)
  are never used. The SLA split is chronological, and workload features only use information
  available at the time.
- **The target is the final `made_sla`.** The flag changes during an incident's life (39% of
  incidents flip), so taking an early value would be wrong.
- **Honest integration.** The two datasets come from different organisations and are linked only
  through priority. The report shows that priority alone predicts SLA breach weakly. Most of the
  signal is routing context such as assignment group and symptom, which the UI lets an agent supply.

## Possible extensions

Fine-tune a multilingual transformer such as `xlm-roberta-base` for triage and compare its
calibration. Add survival analysis (Cox / Weibull AFT) for time-to-resolution. Add conformal
prediction sets for triage. Add drift monitoring on live scores.
