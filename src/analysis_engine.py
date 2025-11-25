import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
import os


class AnalysisEngine:
    """
    Performs the statistical tests described in the AI-HOPE paper.
    """

    @staticmethod
    def perform_case_control(df, case_mask, control_mask, target_col):
        """
        Calculates Odds Ratio for enrichment studies.
        """
        # 1. Define Groups
        case_group = df[case_mask]
        control_group = df[control_mask]

        # 2. Check mutation presence (Target=1 vs Target=0)
        case_pos = case_group[target_col].sum()
        case_neg = len(case_group) - case_pos

        ctrl_pos = control_group[target_col].sum()
        ctrl_neg = len(control_group) - ctrl_pos

        # 3. Fisher's Exact Test
        table = [[case_pos, case_neg], [ctrl_pos, ctrl_neg]]
        odds_ratio, p_value = fisher_exact(table)

        return {
            "test": "Fisher's Exact Test",
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "case_prevalence": f"{case_pos}/{len(case_group)} ({case_pos / len(case_group):.2%})",
            "control_prevalence": f"{ctrl_pos}/{len(control_group)} ({ctrl_pos / len(control_group):.2%})"
        }

    @staticmethod
    def perform_survival_analysis(df, group_col, time_col="OS_MONTHS", event_col="OS_STATUS", output_dir="outputs"):
        """
        Generates Kaplan-Meier curves.
        """
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 6))

        groups = df[group_col].dropna().unique()

        for group in groups:
            mask = (df[group_col] == group)
            T = pd.to_numeric(df[mask][time_col], errors='coerce')
            E = pd.to_numeric(df[mask][event_col], errors='coerce')

            if len(T) > 0:
                kmf.fit(T, event_observed=E, label=str(group))
                kmf.plot_survival_function()

        plt.title(f"Survival Analysis by {group_col}")
        plt.xlabel("Time (Months)")
        plt.ylabel("Survival Probability")

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"survival_{group_col}.png")
        plt.savefig(plot_path)
        plt.close()

        return {
            "plot_path": plot_path,
            "groups_analyzed": list(groups)
        }

    @staticmethod
    def calculate_hazard_ratio(df, group_col, time_col="OS_MONTHS", event_col="OS_STATUS"):
        """
        Calculates the Hazard Ratio using Cox Proportional Hazards.
        """
        try:
            subset = df[[time_col, event_col, group_col]].dropna().copy()

            # Encode group_col to numeric if needed
            if subset[group_col].dtype == 'object':
                subset[group_col] = subset[group_col].astype('category').cat.codes

            cph = CoxPHFitter()
            cph.fit(subset, duration_col=time_col, event_col=event_col)

            summary = cph.summary.iloc[0]  # Get first covariate
            return {
                "hazard_ratio": round(summary['exp(coef)'], 2),
                "ci_lower": round(summary['exp(coef) lower 95%'], 2),
                "ci_upper": round(summary['exp(coef) upper 95%'], 2),
                "p_value": summary['p']
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def perform_global_scan(df, target_col, columns_to_scan):
        """
        Scans all variables to find significant associations with the target.
        """
        significant_findings = []

        for col in columns_to_scan:
            if col == target_col or col not in df.columns:
                continue

            try:
                contingency = pd.crosstab(df[col], df[target_col])
                if contingency.size > 0:
                    # Use Fisher (if 2x2) or Chi-square logic simplified here
                    odds, p_val = fisher_exact(contingency) if contingency.size == 4 else (0, 1.0)

                    if p_val < 0.05:
                        significant_findings.append({
                            "feature": col,
                            "p_value": p_val,
                            "odds_ratio": odds
                        })
            except:
                continue

        return sorted(significant_findings, key=lambda x: x['p_value'])