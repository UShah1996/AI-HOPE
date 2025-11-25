import pandas as pd
import numpy as np
import os

# Create directory
os.makedirs("data/TCGA_COAD", exist_ok=True)

# Generate 200 synthetic samples
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "SampleID": [f"S{i}" for i in range(n)],
    "TUMOR_STAGE": np.random.choice(["Stage I", "Stage II", "Stage III", "Stage IV"], n),
    "KRAS_mutation_status": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    "TP53_Mutation": np.random.choice([0, 1], n, p=[0.5, 0.5]),
    "OS_MONTHS": np.random.exponential(scale=24, size=n).astype(int), # Survival time
    "OS_STATUS": np.random.choice([0, 1], n) # 1=Event (Death), 0=Censored
})

# Save files
data.to_csv("data/TCGA_COAD/main_data.tsv", sep="\t", index=False)
with open("data/TCGA_COAD/index.tsv", "w") as f:
    f.write("\n".join(data.columns))
with open("data/TCGA_COAD/README.txt", "w") as f:
    f.write("Synthetic TCGA Colon Cancer data for AI-HOPE testing.")

print("✅ Data generated in data/TCGA_COAD/")