import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from llm_agent import LLMAgent
from query_parser import QueryParser
from analysis_engine import AnalysisEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
st.set_page_config(page_title="AI-HOPE Agent", layout="wide")


def load_data(dataset_name):
    path = os.path.join(DATA_DIR, dataset_name)
    try:
        data = pd.read_csv(os.path.join(path, "main_data.tsv"), sep="\t")
        with open(os.path.join(path, "index.tsv"), "r") as f:
            columns = [line.strip() for line in f.readlines()]
        return data, columns
    except FileNotFoundError:
        return None, None


st.title("🧬 AI-HOPE: Precision Medicine Agent")
st.markdown("*Locally deployed clinical research assistant [Bioinformatics 2025]*")

if os.path.exists(DATA_DIR):
    available_datasets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
else:
    available_datasets = []

selected_dataset = st.sidebar.selectbox("Choose Cohort", available_datasets)

if selected_dataset:
    df, cols = load_data(selected_dataset)
    if df is not None:
        st.sidebar.success(f"Loaded {len(df)} samples")
        with st.expander("View Data Preview"):
            st.dataframe(df.head())
    else:
        st.error("Dataset files missing.")
        st.stop()
else:
    st.info("Please add a dataset to the 'data/' folder.")
    st.stop()

query = st.text_input("Describe your research question:",
                      placeholder="e.g., Compare survival outcomes in KRAS mutated vs wild-type patients")

if st.button("Analyze"):
    llm = LLMAgent()
    parser = QueryParser()

    with st.spinner("AI-HOPE is thinking..."):

        clarification = llm.check_clarification_needed(query, cols)
        if clarification:
            st.warning("⚠️ Ambiguous Query Detected")
            st.info(f"AI-HOPE needs clarification: **{clarification}**")
            st.stop()

        analysis_category = llm.suggest_analysis(query)
        logic_json = llm.interpret_query(query, cols)

        st.subheader(f"Analysis Category: {analysis_category}")
        with st.expander("See AI Logic (Verified)"):
            st.json(logic_json)

        try:
            cat_str = str(analysis_category).lower()
            json_type = logic_json.get("analysis_type", "").lower()

            final_mode = "unknown"
            if "survival" in cat_str or "survival" in json_type or "1" in cat_str:
                final_mode = "survival"
            elif "case" in cat_str or "control" in cat_str or "2" in cat_str:
                final_mode = "case_control"
            elif "scan" in cat_str or "association" in cat_str or "3" in cat_str:
                final_mode = "scan"

            # --- MODE A: SURVIVAL ANALYSIS ---
            if final_mode == "survival":
                condition = (logic_json.get("group_by") or logic_json.get("target_variable") or logic_json.get(
                    "case_condition"))

                # Try to resolve column
                col = None
                if condition and condition in df.columns:
                    col = condition
                elif condition:
                    parsed_col, _, _ = parser.parse_statement(condition)
                    if parsed_col in df.columns: col = parsed_col

                if col:
                    res = AnalysisEngine.perform_survival_analysis(df, group_col=col)
                    st.image(res['plot_path'])
                    st.write("### Risk Quantification")
                    hr_res = AnalysisEngine.calculate_hazard_ratio(df, group_col=col)
                    if "error" not in hr_res:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Hazard Ratio", hr_res['hazard_ratio'])
                        c2.metric("Confidence Interval", f"{hr_res['ci_lower']} - {hr_res['ci_upper']}")
                        c3.metric("P-Value", f"{hr_res['p_value']:.4f}")
                    else:
                        st.warning(f"Hazard Ratio Error: {hr_res['error']}")
                else:
                    st.error(f"Could not identify a grouping variable. AI logic found: '{condition}'")

            # --- MODE B: CASE-CONTROL STUDY ---
            elif final_mode == "case_control":
                # Check if we have explicit groups (Comparison) or just a target (Prevalence)
                case_raw = logic_json.get("case_condition")
                control_raw = logic_json.get("control_condition")
                target = logic_json.get("target_variable")

                if target and not case_raw:
                    # FALLBACK: Clinical Prevalence (Single variable check)
                    st.info(f"Analyzing Clinical Prevalence for **{target}**")
                    counts = df[target].value_counts()
                    st.bar_chart(counts)
                    st.table(counts)

                elif case_raw:
                    # Standard Comparison
                    case_col, case_op, case_val = parser.parse_statement(case_raw)
                    ctrl_col, ctrl_op, ctrl_val = parser.parse_statement(control_raw)

                    if not ctrl_col and case_col:  # Inverse logic if control missing
                        ctrl_col, ctrl_op = case_col, "not in" if case_op == "in" else "!="
                        ctrl_val = case_val

                    case_mask = parser.apply_filter(df, case_col, case_op, case_val).index
                    ctrl_mask = parser.apply_filter(df, ctrl_col, ctrl_op, ctrl_val).index

                    if target:
                        results = AnalysisEngine.perform_case_control(df, df.index.isin(case_mask),
                                                                      df.index.isin(ctrl_mask), target)
                        col1, col2 = st.columns(2)
                        col1.metric("Odds Ratio", f"{results['odds_ratio']:.2f}")
                        col2.metric("P-Value", f"{results['p_value']:.4f}")
                        st.table(pd.DataFrame({
                            "Metric": ["Case Prevalence", "Control Prevalence"],
                            "Value": [results['case_prevalence'], results['control_prevalence']]
                        }))
                    else:
                        st.error("Target variable not found in query logic.")
                else:
                    st.error("Could not determine analysis parameters from AI logic.")

            # --- MODE C: GLOBAL SCAN ---
            elif final_mode == "scan":
                target = logic_json.get("target_variable")
                if target:
                    st.info(f"Scanning variables for association with **{target}**...")
                    scan_results = AnalysisEngine.perform_global_scan(df, target, cols)
                    if scan_results:
                        st.write("### Significant Associations")
                        st.dataframe(pd.DataFrame(scan_results))
                    else:
                        st.warning("No significant associations found (P < 0.05).")
                else:
                    st.error("Target variable for scan not identified.")

            else:
                st.error(f"Analysis type '{analysis_category}' not recognized.")

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")