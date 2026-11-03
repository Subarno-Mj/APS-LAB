"""Streamlit UI.   Run:  streamlit run app.py"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src import config
from src.pipeline import SupportTriageSystem

st.set_page_config(page_title="Support Triage & SLA Risk", page_icon="🎫", layout="wide")

SAMPLES = {
    "— pick an example —": ("", ""),
    "Outage (EN)": ("Production database unreachable",
                    "Since 9 am our production database cluster is unreachable and all customer-facing "
                    "services are down. We have restarted the nodes without success. Please escalate urgently."),
    "Billing (EN)": ("Charged twice for subscription",
                     "Hello, I was billed twice for my monthly subscription this month. Could you refund "
                     "the duplicate payment and confirm the invoice?"),
    "Product question (EN)": ("Does the tablet support stylus input?",
                              "Before buying I would like to know whether the tablet works with a stylus "
                              "and which accessories are compatible."),
    "Security (DE)": ("Verdacht auf Datenleck",
                      "Sehr geehrtes Support-Team, wir haben ungewöhnliche Zugriffe auf unsere Kundendaten "
                      "festgestellt. Bitte prüfen Sie dringend, ob ein Sicherheitsvorfall vorliegt."),
    "Return (EN)": ("Return a damaged monitor",
                    "The monitor I received has a cracked screen. I would like to return it and get a replacement."),
}


@st.cache_resource
def load_system():
    return SupportTriageSystem()


try:
    system = load_system()
except FileNotFoundError:
    st.error("Models not found. Run `python data_download.py` and `python train.py` first.")
    st.stop()
except Exception as e:  # e.g. models saved with a different scikit-learn version
    st.error(f"Could not load models ({e}). Re-train them with `python train.py`.")
    st.stop()

levels = system.sla.category_levels["open_full"]

st.title("🎫 Probabilistic Support Triage & SLA Risk")
st.caption("Ticket text → calibrated P(queue, priority, type) → abstain-or-route → P(SLA breach) "
           "marginalised over priority uncertainty.")

tab1, tab2, tab3 = st.tabs(["Triage a ticket", "Live SLA re-scoring", "Model report"])

# ------------------------------------------------------------------ tab 1
with tab1:
    pick = st.selectbox("Example", list(SAMPLES))
    s0, b0 = SAMPLES[pick]
    c1, c2 = st.columns([3, 2])
    with c1:
        subject = st.text_input("Subject", s0)
        body = st.text_area("Body", b0, height=180)
    with c2:
        d = st.date_input("Opened on", datetime.now().date())
        tm = st.time_input("Opened at", datetime.now().time().replace(second=0, microsecond=0))
        with st.expander("Optional ITSM context (improves SLA risk a lot)"):
            ctx = {}
            for f in ["assignment_group", "u_symptom", "category", "location", "impact", "urgency"]:
                opts = ["Unknown"] + sorted(v for v in levels.get(f, []) if v not in ("Unknown",))
                ctx[f] = st.selectbox(f, opts, key=f)
            ctx["backlog_open"] = st.number_input("Open tickets in the system now",
                                                  value=int(system.sla.defaults.get("backlog_open", 1450)))

    if st.button("Analyse ticket", type="primary") and body.strip():
        res = system.process(subject, body, datetime.combine(d, tm), ctx)
        tri, sla = res["triage"], res["sla"]

        cols = st.columns(3)
        for col, tgt in zip(cols, config.TRIAGE_TARGETS):
            r = tri[tgt]
            with col:
                st.metric(tgt.capitalize(), r["label"], f"{r['confidence']:.0%} confident",
                          delta_color="normal" if r["auto_route"] else "off")
                st.caption(("✅ auto-route" if r["auto_route"] else "🧑 human review") +
                           f" · threshold {r['threshold']:.2f} · entropy {r['entropy']:.2f}")
                st.bar_chart(pd.Series(r["distribution"]).head(6), horizontal=True, height=200)

        st.divider()
        a, b = st.columns([1, 2])
        with a:
            band_icon = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[sla["risk_band"]]
            st.metric("P(SLA breach)", f"{sla['breach_probability']:.0%}", f"{band_icon} {sla['risk_band']}",
                      delta_color="off")
            st.progress(min(sla["breach_probability"], 1.0))
            st.caption(f"model: `{sla['model']}` · ± {sla['uncertainty_from_priority']:.2f} from priority uncertainty")
        with b:
            st.markdown("**How the risk is composed**  \nP(breach) = Σ P(priority | text) · P(breach | priority, context)")
            st.dataframe(pd.DataFrame(sla["conditional_risk"]).style.format(
                {"p_priority": "{:.2f}", "p_breach_given_priority": "{:.2f}"}), hide_index=True)
        st.subheader("Recommended actions")
        for x in res["actions"]:
            st.markdown(f"- {x}")
        with st.expander("Raw output"):
            st.json(res)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Re-score an open incident at any update (trained on ServiceNow event log, only events still within SLA).")
    c = st.columns(3)
    rec = {
        "priority": c[0].selectbox("Priority", config.ITSM_PRIORITIES, index=2),
        "incident_state": c[0].selectbox("State", ["New", "Active", "Awaiting User Info", "Awaiting Problem"]),
        "elapsed_h": c[1].number_input("Hours since opened", 0.0, 2000.0, 4.0),
        "reassignment_count": c[1].number_input("Reassignments", 0, 30, 0),
        "reopen_count": c[2].number_input("Reopens", 0, 10, 0),
        "sys_mod_count": c[2].number_input("Updates so far", 0, 100, 2),
        "assignment_group": st.selectbox("Assignment group", ["Unknown"] + sorted(levels.get("assignment_group", []))),
    }
    rec["current_group"] = rec["assignment_group"]
    if st.button("Re-score"):
        r = system.rescore_live(rec)
        st.metric("P(SLA breach)", f"{r['breach_probability']:.0%}", r["risk_band"], delta_color="off")

# ------------------------------------------------------------------ tab 3
with tab3:
    mpath = config.REPORTS / "metrics.json"
    if mpath.exists():
        md = (config.REPORTS / "RESULTS.md").read_text(encoding="utf-8")
        for block in md.split("\n\n"):
            if block.startswith("![") and "(" in block:
                img = config.REPORTS / block[block.index("(") + 1: block.index(")")]
                if img.exists():
                    st.image(str(img))
            else:
                st.markdown(block)
    else:
        st.info("Run `python train.py` to generate the report.")
