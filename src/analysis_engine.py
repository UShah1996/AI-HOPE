import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
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
        Uses Fisher's exact test for 2x2 tables and chi-square for larger tables.
        """
        significant_findings = []

        for col in columns_to_scan:
            if col == target_col or col not in df.columns:
                continue

            try:
                # Create contingency table
                contingency = pd.crosstab(df[col], df[target_col])
                
                # Remove rows/columns with all zeros
                contingency = contingency.loc[(contingency != 0).any(axis=1), (contingency != 0).any(axis=0)]
                
                if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                
                # For 2x2 tables, use Fisher's exact test
                if contingency.size == 4 and contingency.shape == (2, 2):
                    odds, p_val = fisher_exact(contingency)
                    test_stat = odds
                    test_name = "Fisher's Exact"
                # For larger tables, use chi-square test
                else:
                    # Check if expected frequencies are sufficient (all >= 5 for chi-square)
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    # If expected frequencies are too small, use Fisher's exact (if table is small enough)
                    if expected.min() < 5 and contingency.size <= 20:
                        # For small tables, try Fisher's exact
                        try:
                            odds, p_val = fisher_exact(contingency)
                            test_stat = odds
                            test_name = "Fisher's Exact"
                        except:
                            # Fall back to chi-square even if expected < 5
                            test_stat = chi2
                            test_name = "Chi-square"
                    else:
                        test_stat = chi2
                        test_name = "Chi-square"
                
                # Calculate effect size (Cramér's V for larger tables, OR for 2x2)
                if contingency.size == 4 and contingency.shape == (2, 2):
                    effect_size = odds
                    effect_label = "Odds Ratio"
                else:
                    # Cramér's V for larger tables
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    if min_dim > 0 and n > 0:
                        # Get chi2 value (may have been calculated above)
                        if 'chi2' in locals() and chi2 > 0:
                            cramers_v = np.sqrt(chi2 / (n * min_dim))
                        else:
                            # Recalculate chi-square if needed
                            chi2_recalc, _, _, _ = chi2_contingency(contingency)
                            cramers_v = np.sqrt(chi2_recalc / (n * min_dim)) if chi2_recalc > 0 else 0
                        effect_size = cramers_v
                        effect_label = "Cramér's V"
                    else:
                        effect_size = 0
                        effect_label = "N/A"

                if p_val < 0.05:
                    significant_findings.append({
                        "feature": col,
                        "p_value": round(p_val, 4),
                        "effect_size": round(effect_size, 4),
                        "effect_label": effect_label,
                        "test": test_name
                    })
            except Exception as e:
                # Skip variables that can't be analyzed
                continue

        return sorted(significant_findings, key=lambda x: x['p_value'])