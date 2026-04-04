"""
streamlit_app/pages/4_Evaluation.py
─────────────────────────────────────
Run Ragas benchmark and display results.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Evaluation — RAG Assistant", page_icon="📊", layout="wide")

from streamlit_app.utils.session import init_session
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.utils import api_client

init_session()
render_sidebar()

st.title("📊 Evaluation")

st.markdown(
    "Run the **Ragas** benchmark suite against the pre-loaded question set "
    "(`evaluation/benchmark_questions.json`)."
)

st.subheader("Evaluation Mode")

mode = st.radio(
    "Choose evaluation mode",
    options=["fast", "full"],
    format_func=lambda x: "Fast evaluation" if x == "fast" else "Full evaluation",
    help=(
        "Fast mode runs all 5 questions with 2 metrics "
        "(faithfulness, answer_relevancy). "
        "Full mode runs all benchmark questions with 4 metrics "
        "(faithfulness, answer_relevancy, context_recall, context_precision)."
    ),
)

if mode == "fast":
    st.info(
        "Fast mode: all 5 questions, 2 metrics.\n\n"
        "- Faithfulness\n"
        "- Answer relevancy"
    )
    st.caption("Estimated runtime: ~1–2 minutes on local Ollama CPU inference.")
else:
    st.info(
        "Full mode: all questions, 4 metrics.\n\n"
        "- Faithfulness\n"
        "- Answer relevancy\n"
        "- Context recall\n"
        "- Context precision"
    )
    st.caption("Estimated runtime: can be much longer depending on model and metrics (~5-10m).")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running Ragas evaluation… this may take a while."):
            try:
                result = api_client.run_evaluation(mode=mode)
                st.success(f"✅ Run `{result['run_id']}` completed!")

                st.json({
                    "mode": result["mode"],
                    "num_questions": result["num_questions"],
                    "metrics": result["metrics"],
                    "started_at": result["started_at"],
                    "finished_at": result["finished_at"],
                })
            except Exception as e:
                st.error("Evaluation failed.")
                st.exception(e)

st.divider()
st.subheader("Past Evaluation Runs")
try:
    data = api_client.get_evaluation_results()
    results = data.get("results", [])
    if not results:
        st.info("No evaluation runs yet.")
    else:
        for run in reversed(results):
            mode_label = run.get("mode", "unknown")
            with st.expander(f"Run `{run['run_id']}` — {mode_label} — {run['started_at'][:10]}"):
                st.write(f"**Mode:** {run.get('mode', 'N/A')}")
                st.write(f"**Questions evaluated:** {run.get('num_questions', 0)}")
                
                m = run.get("metrics", {})
                cols = st.columns(len(m))
                for col, (metric, val) in zip(cols, m.items()):
                    col.metric(metric.replace("_", " ").title(), f"{val:.3f}")
except Exception as e:
    st.error(f"Could not fetch results: {e}")
