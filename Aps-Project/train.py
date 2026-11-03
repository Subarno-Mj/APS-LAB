"""Train and evaluate the whole system, then write models/ and reports/.

    python train.py              # everything (~5-10 min on a laptop CPU)
    python train.py --skip-triage
    python train.py --skip-sla
"""
import argparse
import time

from src import config
from src.reports import sla_figures, triage_figures, write_markdown
from src.sla_model import SLAModel, build_tables
from src.triage_model import TriageModel, load_tickets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-triage", action="store_true")
    ap.add_argument("--skip-sla", action="store_true")
    args = ap.parse_args()

    for f in (config.TICKETS_FILE, config.INCIDENTS_FILE):
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run: python data_download.py")

    report = {}
    t0 = time.time()
    if not args.skip_triage:
        print("== Triage: loading tickets")
        df = load_tickets()
        print(f"   {len(df):,} tickets, languages: {df['language'].value_counts().to_dict()}")
        triage = TriageModel()
        report["triage"] = triage.fit_evaluate(df)

        raw = {t: tm.metrics["_test"] for t, tm in triage.targets.items()}
        triage_figures(triage, raw)
        triage.save()
        print(f"   saved {config.MODELS / 'triage_model.joblib'}  ({time.time() - t0:.0f}s)")
    else:
        triage = TriageModel.load()

    if not args.skip_sla:
        print("== SLA: building incident tables")
        inc, live = build_tables()
        print(f"   {len(inc):,} incidents, {len(live):,} live scoring events, breach rate {inc['breach'].mean():.1%}")
        sla = SLAModel()
        report["sla"] = sla.fit_evaluate(inc, live)
        sla_figures(sla)
        sla.save()
        print(f"   saved {config.MODELS / 'sla_model.joblib'}  ({time.time() - t0:.0f}s)")
    else:
        sla = SLAModel.load()

    if "triage" in report and "sla" in report:
        write_markdown(report, triage, sla)
        print(f"== Report: {config.REPORTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
