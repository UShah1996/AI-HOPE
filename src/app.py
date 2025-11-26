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
    """
    Loads the 3-file format: README, Index, Data Table.
    """
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
st.sidebar.header("Dataset Selection")
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
        st.error("Dataset files missing (Requires main_data.tsv and index.tsv)")
        st.stop()
else:
    st.info("Please add a dataset to the 'data/' folder.")
    st.stop()

# Main Chat Interface
query = st.text_input("Describe your research question:",
                      placeholder="e.g., Tell me everything associated with overall survival")

if st.button("Analyze"):
    # 1. Initialize Agents
    llm = LLMAgent()
    parser = QueryParser()

    with st.spinner("AI-HOPE is thinking..."):
        # 2. Determine Intent & Logic
        # Note: We ask the LLM to classify into 3 specific buckets now
        suggested_type = llm.suggest_analysis(query)
        logic_json = llm.interpret_query(query, cols)
        
        # Check for explicit association scan keywords in query (highest priority)
        query_lower = query.lower()
        association_keywords = ["everything associated", "tell me everything", "what variables", "what is associated", 
                               "global scan", "scan for", "find associations", "all associations"]
        
        if any(keyword in query_lower for keyword in association_keywords):
            analysis_type = "association scan"
        # Prioritize logic_json analysis_type over suggested_type (more accurate)
        else:
            analysis_type = logic_json.get("analysis_type", suggested_type).lower()
            if not analysis_type or analysis_type == "error":
                analysis_type = suggested_type.lower()

        st.subheader(f"Analysis Type: {analysis_type.title()}")

        # Display the interpreted logic for transparency (Explainable AI)
        with st.expander("See AI Logic"):
            st.json(logic_json)
            # If there's an error, show the raw content for debugging
            if "error" in logic_json and "raw_content" in logic_json:
                st.error("LLM Response Error")
                st.text_area("Raw LLM Response:", logic_json.get("raw_content", "No content"), height=100)

        # 3. Execute Analysis based on Type
        try:
            # --- MODE A: SURVIVAL ANALYSIS ---
            if "survival" in analysis_type:
                # Outcome variables should never be used as grouping variables
                outcome_vars = ["OS_STATUS", "OS_MONTHS", "SampleID"]
                
                # Try multiple fields from logic_json to find grouping variable
                condition = (logic_json.get("grouping_variable") or 
                            logic_json.get("predictor") or
                            logic_json.get("case_condition") or 
                            logic_json.get("target_variable"))
                
                col, op, val = parser.parse_statement(condition)
                
                # If parsing failed but condition is a valid column name, use it directly
                if not col and condition and condition.strip() in df.columns:
                    col = condition.strip()
                
                # Validate: don't use outcome variables as grouping variables
                if col in outcome_vars:
                    # Try to find a valid grouping variable from other columns
                    # Look for mutation or stage variables mentioned in the query
                    query_lower = query.lower()
                    found_valid_col = False
                    for potential_col in df.columns:
                        if (potential_col not in outcome_vars and 
                            potential_col.lower() in query_lower):
                            col = potential_col
                            found_valid_col = True
                            break
                    
                    # If still an outcome variable, show error
                    if not found_valid_col or col in outcome_vars:
                        st.error(f"Cannot use '{col}' as a grouping variable. It is an outcome variable. Please specify a grouping variable like TP53_Mutation, KRAS_mutation_status, or TUMOR_STAGE.")
                        col = None

                if col and col in df.columns:
                    # A. Kaplan-Meier Curve
                    res = AnalysisEngine.perform_survival_analysis(df, group_col=col)
                    st.image(res['plot_path'])

                    # B. Hazard Ratio (New Feature)
                    st.write("### Risk Quantification")
                    hr_res = AnalysisEngine.calculate_hazard_ratio(df, group_col=col)

                    if "error" not in hr_res:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Hazard Ratio", hr_res['hazard_ratio'])
                        col2.metric("Confidence Interval", f"{hr_res['ci_lower']} - {hr_res['ci_upper']}")
                        col3.metric("P-Value", f"{hr_res['p_value']:.4f}")
                        st.caption(
                            f"Interpretation: Patients in this group have {hr_res['hazard_ratio']}x the risk compared to baseline.")
                    else:
                        st.warning(f"Could not calculate Hazard Ratio: {hr_res['error']}")
                else:
                    st.error(f"Could not identify a valid grouping variable in query: '{condition}'")

            # --- MODE B: CASE-CONTROL STUDY ---
            elif "case" in analysis_type or "control" in analysis_type:
                # Parse Cohorts
                case_col, case_op, case_val = parser.parse_statement(logic_json.get("case_condition", ""))
                ctrl_col, ctrl_op, ctrl_val = parser.parse_statement(logic_json.get("control_condition", ""))

                # Apply Filters
                case_mask = parser.apply_filter(df, case_col, case_op, case_val).index
                ctrl_mask = parser.apply_filter(df, ctrl_col, ctrl_op, ctrl_val).index

                # Run Stats
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
                    st.error("Target variable not found in query.")

            # --- MODE C: GLOBAL DISCOVERY SCAN (New Feature) ---
            elif "scan" in analysis_type or "association" in analysis_type:
                target = logic_json.get("target_variable")
                if target and target in df.columns:
                    st.info(f"Scanning all variables for association with **{target}**...")

                    # Use actual dataframe columns, excluding the target and outcome variables
                    columns_to_scan = [col for col in df.columns 
                                      if col != target 
                                      and col not in ["SampleID", "OS_STATUS", "OS_MONTHS"]]
                    
                    # Scan all columns - get all results to show what was tested
                    all_results = AnalysisEngine.perform_global_scan(df, target, columns_to_scan, return_all=True)
                    significant_results = [r for r in all_results if r['p_value'] < 0.05]

                    if significant_results:
                        st.write("### Significant Associations (P < 0.05)")
                        scan_df = pd.DataFrame(significant_results)
                        # Create a readable effect size column
                        if 'effect_size' in scan_df.columns and 'effect_label' in scan_df.columns:
                            scan_df['Effect'] = scan_df.apply(
                                lambda row: f"{row['effect_size']:.4f} ({row['effect_label']})", axis=1
                            )
                            display_cols = ['feature', 'p_value', 'Effect']
                        else:
                            display_cols = ['feature', 'p_value']
                        if 'test' in scan_df.columns:
                            display_cols.append('test')
                        
                        st.dataframe(scan_df[display_cols].style.highlight_min(subset=['p_value'], color='lightgreen'))
                    else:
                        st.warning(f"No significant associations found (p < 0.05) between **{target}** and {len(columns_to_scan)} tested variables.")
                    
                    # Show all results in an expander for transparency
                    if all_results:
                        with st.expander("View All Tested Associations"):
                            all_df = pd.DataFrame(all_results)
                            if 'effect_size' in all_df.columns and 'effect_label' in all_df.columns:
                                all_df['Effect'] = all_df.apply(
                                    lambda row: f"{row['effect_size']:.4f} ({row['effect_label']})", axis=1
                                )
                                display_cols = ['feature', 'p_value', 'Effect']
                            else:
                                display_cols = ['feature', 'p_value']
                            if 'test' in all_df.columns:
                                display_cols.append('test')
                            
                            # Highlight significant ones
                            st.dataframe(
                                all_df[display_cols].style.apply(
                                    lambda row: ['background-color: lightgreen' if row['p_value'] < 0.05 else '' for _ in row],
                                    axis=1
                                )
                            )
                    else:
                        st.caption(f"Tested {len(columns_to_scan)} variables: {', '.join(columns_to_scan)}")
                else:
                    st.error(f"Target variable '{target}' not found in dataset. Available columns: {', '.join(df.columns.tolist())}")

            else:
                st.warning("Analysis type not recognized. Please try a specific question like 'Compare X vs Y'.")

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.info("Tip: Ensure your query refers to columns that exist in the dataset.")