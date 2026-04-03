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

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running Ragas evaluation… this may take a few minutes."):
            try:
                result = api_client.run_evaluation()
                st.success(f"✅ Run `{result['run_id']}` completed!")
                st.json({
                    "num_questions": result["num_questions"],
                    "metrics": result["metrics"],
                    "started_at": result["started_at"],
                    "finished_at": result["finished_at"],
                })
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

st.divider()
st.subheader("Past Evaluation Runs")
try:
    data = api_client.get_evaluation_results()
    results = data.get("results", [])
    if not results:
        st.info("No evaluation runs yet.")
    else:
        for run in reversed(results):
            with st.expander(f"Run `{run['run_id']}` — {run['started_at'][:10]}"):
                m = run.get("metrics", {})
                cols = st.columns(len(m))
                for col, (metric, val) in zip(cols, m.items()):
                    col.metric(metric.replace("_", " ").title(), f"{val:.3f}")
except Exception as e:
    st.error(f"Could not fetch results: {e}")
