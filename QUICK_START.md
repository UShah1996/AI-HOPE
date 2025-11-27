# Quick Start Guide - Running AI-HOPE with Safety Layer

## Step-by-Step: Running the App

### 1. Check Prerequisites
```bash
# Make sure Ollama is running (it usually auto-starts)
ollama list

# If you don't have llama3, pull it:
ollama pull llama3
```

### 2. Start the Streamlit App
```bash
# From the project root directory (where you are now)
streamlit run src/app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 3. Select Dataset
- In the left sidebar, select **"TCGA_COAD"** from the dropdown
- You should see: "Loaded X samples" message
- Expand "View Data Preview" to see the data

---

## Test Cases

### ✅ VALID Test Cases (Should Work)

#### Test 1: Survival Analysis
**Query:**
```
Compare survival outcomes for patients with TP53_Mutation
```

**Expected Result:**
- Safety layer passes ✅
- You see a survival plot (Kaplan-Meier curve)
- Hazard ratio metrics displayed

---

#### Test 2: Case-Control Study
**Query:**
```
Compare TP53_Mutation frequency in Stage III vs Stage I
```

**Expected Result:**
- Safety layer passes ✅
- Odds ratio and p-value displayed
- Prevalence table shown

---

#### Test 3: Association Scan
**Query:**
```
Tell me everything associated with TUMOR_STAGE
```

**Expected Result:**
- Safety layer passes ✅
- List of significant associations (p < 0.05)
- All tested variables shown in expander

---

#### Test 4: Another Survival Query
**Query:**
```
Does KRAS_mutation_status affect overall survival?
```

**Expected Result:**
- Safety layer passes ✅
- Survival analysis runs successfully

---

### ❌ INVALID Test Cases (Should Show Safety Errors)

#### Test 5: Invalid Column Name - Survival
**Query:**
```
Compare survival outcomes for patients with FAKE_COLUMN_NAME
```

**Expected Result:**
- ❌ Red error message appears:
  ```
  Safety check failed: Grouping variable 'FAKE_COLUMN_NAME' not found in dataset. 
  Available columns: KRAS_mutation_status, OS_MONTHS, OS_STATUS, SampleID, TP53_Mutation, TUMOR_STAGE
  ```
- Analysis does NOT run
- No plot or results shown

---

#### Test 6: Invalid Target - Association Scan
**Query:**
```
Tell me everything associated with INVALID_TARGET_COLUMN
```

**Expected Result:**
- ❌ Error message:
  ```
  Safety check failed: Target variable 'INVALID_TARGET_COLUMN' not found in dataset...
  ```
- Analysis does NOT run

---

#### Test 7: Invalid Target - Case-Control
**Query:**
```
Compare FAKE_TARGET frequency in Stage III vs Stage I
```

**Expected Result:**
- ❌ Error message about missing target variable
- Analysis does NOT run

---

#### Test 8: Missing Grouping Variable
**Query:**
```
Compare survival outcomes
```

**Expected Result:**
- ❌ Error message:
  ```
  Safety check failed: Survival analysis requires a grouping variable...
  ```
- Analysis does NOT run

---

## What to Look For

### ✅ Success Indicators:
1. **Valid queries**: Analysis runs normally, you see plots/results
2. **Invalid queries**: Red error box appears BEFORE any analysis runs
3. **Error messages**: Clear, user-friendly, show available columns
4. **No crashes**: App handles errors gracefully

### 🔍 Debugging Tips:
- Expand **"See AI Logic"** to see what the LLM generated
- Check the terminal for any Python errors
- If Ollama errors occur, make sure it's running: `ollama serve`

---

## Quick Command Reference

```bash
# Run the app
streamlit run src/app.py

# Test safety layer directly (no Ollama needed)
python test_safety_layer.py

# Check Ollama status
ollama list

# Pull llama3 model (if needed)
ollama pull llama3
```

---

## Available Columns in TCGA_COAD Dataset

When testing, use these valid column names:
- `SampleID`
- `TUMOR_STAGE`
- `KRAS_mutation_status`
- `TP53_Mutation`
- `OS_MONTHS`
- `OS_STATUS`

Use any other name to trigger a safety layer error!

