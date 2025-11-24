# src/analysis_engine.py
import pandas as pd
from scipy import stats
from lifelines import statistics
from visualization import plot_kaplan_meier, plot_contingency_heatmap

def run_analysis(df: pd.DataFrame, intent):
    print(f"🧠 Running {intent.query_type} analysis...")
    try:
        if intent.query_type == "association":
            result = association_analysis(df, intent)
        elif intent.query_type == "survival":
            result = survival_analysis(df, intent)
        else:
            result = "⚠️ Unsupported query type."
    except Exception as e:
        result = f"❌ Analysis failed: {e}"
    print(result)
    return result


def association_analysis(df: pd.DataFrame, intent):
    # column hygiene
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    for col in (intent.target_variable, intent.group_variable):
        if col not in df.columns:
            return f"❌ Missing column '{col}'. Available: {list(df.columns)}"

    x = df[intent.target_variable]
    y = df[intent.group_variable]
    table = pd.crosstab(x, y)

    if table.shape == (2, 2):
        oddsratio, p = stats.fisher_exact(table)
        plot_path = plot_contingency_heatmap(df, intent.target_variable, intent.group_variable)
        return f"Fisher’s exact test OR={oddsratio:.3f}, p={p:.4f}. Plot: {plot_path}"
    else:
        chi2, p, dof, expected = stats.chi2_contingency(table)
        plot_path = plot_contingency_heatmap(df, intent.target_variable, intent.group_variable)
        return f"Chi-square test χ²={chi2:.3f}, p={p:.4f}. Plot: {plot_path}"


def survival_analysis(df: pd.DataFrame, intent):
    # column hygiene
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    time_col, event_col = "OS_MONTHS", "OS_EVENT"
    group_col = intent.target_variable  # e.g., KRAS_mutation_status

    # Validate required columns
    missing = [c for c in (group_col, time_col, event_col) if c not in df.columns]
    if missing:
        return f"❌ Missing required columns: {missing}. Available: {list(df.columns)}"

    # If user intent includes a treatment variable like FOLFOX_treatment,
    # restrict to treated patients (==1) to mirror the paper’s example.
    if intent.group_variable and intent.group_variable in df.columns:
        if df[intent.group_variable].dropna().nunique() <= 2:
            # convention: 1 = treated
            df = df[df[intent.group_variable] == 1]

    # Need two groups to compare
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return f"❌ Need at least two groups in '{group_col}' for survival comparison. Found: {groups}"

    path = plot_kaplan_meier(df, group_col, time_col, event_col)

    if len(groups) == 2:
        g1 = df[df[group_col] == groups[0]]
        g2 = df[df[group_col] == groups[1]]
        result = statistics.logrank_test(
            g1[time_col], g2[time_col],
            g1[event_col], g2[event_col]
        )
        return f"Log-rank test p={result.p_value:.4f}. KM saved: {path}"

    return f"KM saved: {path} (log-rank requires exactly 2 groups; found {len(groups)})"
