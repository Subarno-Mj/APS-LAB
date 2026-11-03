"""Command-line demo.

    python predict.py --subject "Server down" --body "Our main server is not responding since morning..."
    python predict.py --subject "..." --body "..." --group "Group 70" --symptom "Symptom 491"
"""
import argparse
import json

from src.pipeline import SupportTriageSystem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="")
    ap.add_argument("--body", required=True)
    ap.add_argument("--group", help="ITSM assignment group, e.g. 'Group 70'")
    ap.add_argument("--symptom", help="ITSM u_symptom, e.g. 'Symptom 491'")
    ap.add_argument("--json", action="store_true", help="print full JSON")
    a = ap.parse_args()

    ctx = {"assignment_group": a.group, "u_symptom": a.symptom}
    res = SupportTriageSystem().process(a.subject, a.body, context=ctx)
    if a.json:
        print(json.dumps(res, indent=2))
        return
    t, s = res["triage"], res["sla"]
    for k in ("queue", "priority", "type"):
        flag = "auto" if t[k]["auto_route"] else "HUMAN REVIEW"
        print(f"{k:9s}: {t[k]['label']:32s} p={t[k]['confidence']:.2f}  [{flag}]")
    print(f"SLA risk : P(breach)={s['breach_probability']:.2f}  band={s['risk_band']}  "
          f"(+/-{s['uncertainty_from_priority']:.2f} from priority uncertainty, model={s['model']})")
    for x in res["actions"]:
        print(" ->", x)


if __name__ == "__main__":
    main()
