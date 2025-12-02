import streamlit as st
import pandas as pd
import os
from llm_agent import LLMAgent
from query_parser import QueryParser
from analysis_engine import AnalysisEngine

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
st.set_page_config(page_title="AI-HOPE Agent", layout="wide")


# --- Helper Functions ---
def load_data(dataset_name):
    path = os.path.join(DATA_DIR, dataset_name)
    try:
        data = pd.read_csv(os.path.join(path, "main_data.tsv"), sep="\t")
        with open(os.path.join(path, "index.tsv"), "r") as f:
            columns = [line.strip() for line in f.readlines()]
        return data, columns
    except FileNotFoundError:
        return None, None


# --- Main App Interface ---
st.title("🧬 AI-HOPE: Precision Medicine Agent")
st.markdown("*Locally deployed clinical research assistant [Bioinformatics 2025]*")

# Sidebar: Data Selection
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

# Main Chat Interface
query = st.text_input("Describe your research question:",
                      placeholder="e.g., Compare survival outcomes in KRAS mutated vs wild-type patients")

if st.button("Analyze"):
    llm = LLMAgent()
    parser = QueryParser()

    with st.spinner("AI-HOPE is thinking..."):

        # 1. Validation Check
        # We relaxed the Clarifier in llm_agent.py, but if it still triggers, show warning.
        clarification = llm.check_clarification_needed(query, cols)

        if clarification:
            st.warning("⚠️ Ambiguous Query Detected")
            st.info(f"AI-HOPE needs clarification: **{clarification}**")
            st.stop()

            # 2. Logic Generation
        analysis_type = llm.suggest_analysis(query)
        logic_json = llm.interpret_query(query, cols)

        st.subheader(f"Analysis Type: {analysis_type}")
        with st.expander("See AI Logic (Verified)"):
            st.json(logic_json)

        try:
            # --- ROBUST LOGIC ROUTING (Fixes "Analysis Not Recognized" Errors) ---

            # Convert to lower case for easy matching
            atype = analysis_type.lower()

            # MODE A: SURVIVAL ANALYSIS
            if "survival" in atype:
                # Look for ANY condition that acts as a group
                condition = (
                        logic_json.get("group_by") or
                        logic_json.get("case_condition") or
                        logic_json.get("target_variable")
                )

                # Try to resolve the column name
                col = None
                if condition:
                    if condition in df.columns:
                        col = condition
                    else:
                        # Try parsing "Column is Value"
                        parsed_col, _, _ = parser.parse_statement(condition)
                        if parsed_col in df.columns:
                            col = parsed_col

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
                    st.error(f"Could not identify grouping variable. Found condition: '{condition}'")

            # MODE B: CASE-CONTROL STUDY
            elif "case" in atype and "control" in atype:
                case_col, case_op, case_val = parser.parse_statement(logic_json.get("case_condition", ""))
                ctrl_col, ctrl_op, ctrl_val = parser.parse_statement(logic_json.get("control_condition", ""))

                case_mask = parser.apply_filter(df, case_col, case_op, case_val).index
                ctrl_mask = parser.apply_filter(df, ctrl_col, ctrl_op, ctrl_val).index

                target = logic_json.get("target_variable")
                if target:
                    mask_c = df.index.isin(case_mask)
                    mask_ct = df.index.isin(ctrl_mask)
                    results = AnalysisEngine.perform_case_control(df, mask_c, mask_ct, target)

                    col1, col2 = st.columns(2)
                    col1.metric("Odds Ratio", f"{results['odds_ratio']:.2f}")
                    col2.metric("P-Value", f"{results['p_value']:.4f}")
                    st.table(pd.DataFrame({
                        "Metric": ["Case Prevalence", "Control Prevalence"],
                        "Value": [results['case_prevalence'], results['control_prevalence']]
                    }))
                else:
                    st.error("Target variable not found in query logic.")

            # MODE C: GLOBAL SCAN
            elif "scan" in atype or "association" in atype:
                target = logic_json.get("target_variable")
                if target:
                    st.info(f"Scanning all variables for association with **{target}**...")
                    scan_results = AnalysisEngine.perform_global_scan(df, target, cols)

                    if scan_results:
                        st.write("### Significant Associations (P < 0.05)")
                        st.dataframe(
                            pd.DataFrame(scan_results).style.highlight_min(subset=['p_value'], color='lightgreen'))
                    else:
                        st.warning("No significant associations found.")
                else:
                    st.error("Target variable for scan not identified.")

            else:
                st.warning(f"Analysis type '{analysis_type}' not recognized. Please try rephrasing.")

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")