# AI-HOPE: AI-Driven Agent for Precision Medicine

**An open-source implementation of the AI-HOPE system described in *Bioinformatics (2025)*.**

This repository contains the implementation of **AI-HOPE (Artificial Intelligence Agent for High-Optimization and Precision Medicine)**, an LLM-driven system designed to integrate clinical and genomic data through natural language interactions.

## 📖 Overview

The growing complexity of clinical cancer research requires tools that can bridge the gap between complex data and researchers without programming expertise. AI-HOPE addresses this by allowing domain experts to conduct integrative data analyses—such as survival analysis and association studies—using simple conversational queries.

### Core Philosophy
* **Natural Language Interface:** Users provide instructions in plain English (e.g., "Compare survival outcomes between groups").
* **Privacy-First Architecture:** Operates as a **closed system** using locally deployed Large Language Models (LLMs) to ensure HIPAA and GDPR compliance by preventing data leakage.
* **Automated Statistics:** Automatically selects and executes statistical tests (Odds Ratios, Log-rank tests, Hazard Ratios) based on the user's intent.

---

## 🏗️ System Architecture

This implementation follows the three-stage workflow described in the original paper:

1.  **Conversational Setup (GUI):** A user interacts with the system via a Graphical User Interface (GUI), submitting natural language queries.
2.  **Logic Extraction (The "Brain"):** A local LLM (Llama3) acts as a reasoning agent. It does *not* analyze the data directly; instead, it interprets user instructions and converts them into executable logic (e.g., parsing "Age > 50" into a filter operation).
3.  **Automated Analysis (The Engine):** The system executes the generated logic against local data files to perform prevalence testing, association analysis, and survival modeling.

---

## 📂 Data Formatting Requirements

To ensure the agent can autonomously read your data, datasets must be organized into specific folders containing **three mandatory components**:

1.  **`README.txt`**: A text file providing an overview of the dataset.
2.  **`index.txt`**: A list of key attributes (column headers) available for analysis.
3.  **`data_table.tsv`**: The main tab-delimited data table where rows represent samples and columns represent attributes.

**Directory Structure Example:**

```text
data/
└── your_dataset_name/
    ├── README.txt
    ├── index.txt
    └── data_table.tsv
```

## 🛠️ Installation & Setup
Prerequisites
Python 3.9+
Ollama: This project requires a local instance of Llama3. Download Ollama from ollama.com.


1. Clone the Repository
    Bash: git clone [https://github.com/UShah1996/AI-HOPE.git](https://github.com/UShah1996/AI-HOPE.git)
    cd ai-hope-implementation
2. Install Python Dependencies
    Bash: pip install -r requirements.txt
3. Initialize Local LLM
Ensure Ollama is running and pull the Llama3 model (or the specific model version you intend to use).
    Bash: ollama run llama3
4. Run the Application
    Bash: streamlit run src/app.py

## 🧪 Capabilities & Usage
AI-HOPE supports two primary modes of analysis triggered by natural language:

1. Case-Control Studies
Define cohorts based on clinical criteria and compare them.
Example Query: "Does the frequency of TP53 mutations differ between early- and late-stage CRC?".
Mechanism: The system defines "Case" (Late Stage) and "Control" (Early Stage) groups using logical expressions like TUMOR_STAGE is in {T3, T4} and performs an Odds Ratio test .

2. Survival Analysis
Compare outcomes between groups using Kaplan-Meier curves and Hazard Ratios.
Example Query: "Compare survival outcomes between FOLFOX-treated patients with and without KRAS mutations.".
Mechanism: The system filters for treated patients, stratifies by mutation status (KRAS_mutation_status is 1 vs 0), and computes progression-free survival statistics .

3. Global Association Scans
Identify all variables significantly associated with a specific outcome.
Example Query: "Tell me everything associated with overall survival in colon cancer.".
Mechanism: The agent scans all available variables in the index.txt to identify significant associations.

## 🛡️ Privacy Note
This software is designed for local deployment only. To maintain the security of sensitive clinical data, do not modify the code to send data to external APIs (e.g., OpenAI, Anthropic). The logic extraction is handled entirely by the local Llama3 instance to avoid online data exchange.


## 📚 Reference
This implementation is based on:

AI-HOPE: an AI-driven conversational agent for enhanced clinical and genomic data integration in precision medicine research Bioinformatics, 2025, 41(7), btaf359. https://doi.org/10.1093/bioinformatics/btaf359.