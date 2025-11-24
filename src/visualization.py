import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_kaplan_meier(df, group_col, time_col="OS_MONTHS", event_col="OS_EVENT"):
    """Draw Kaplan–Meier survival curve."""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7, 5))

    for group in df[group_col].unique():
        mask = df[group_col] == group
        kmf.fit(df[mask][time_col], df[mask][event_col], label=f"{group_col}={group}")
        kmf.plot_survival_function(ci_show=True)

    plt.title(f"Survival curves by {group_col}")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"km_{group_col}.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_contingency_heatmap(df, col1, col2):
    """Generate a heatmap for cross-tabulated categorical data."""
    table = df.groupby([col1, col2]).size().unstack(fill_value=0)
    plt.figure(figsize=(5, 4))
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{col1} vs {col2} counts")
    path = os.path.join(OUTPUT)
