# Testing Guide for Safety Layer

## Quick Start

### 1. Prerequisites Check
```bash
# Make sure you have Python 3.9+
python --version

# Make sure Ollama is installed and running
ollama --version

# Make sure you have the llama3 model
ollama list
# If llama3 is not listed, run: ollama pull llama3
```

### 2. Install Dependencies
```bash
# Activate your virtual environment (if using one)
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
```bash
# From the project root directory
streamlit run src/app.py
```

The app should open in your browser at `http://localhost:8501`

---

## Testing the Safety Layer

### Test Setup
1. In the Streamlit app sidebar, select **"TCGA_COAD"** dataset
2. You should see "Loaded X samples" message

### Test Cases

#### ✅ Test 1: Valid Survival Analysis (Should Pass)
**Query:** `"Compare survival outcomes for patients with TP53_Mutation"`

**Expected:** 
- Safety layer passes validation
- Analysis executes successfully
- You see a survival plot

---

#### ❌ Test 2: Invalid Column Name (Should Fail)
**Query:** `"Compare survival outcomes for patients with INVALID_COLUMN"`

**Expected:**
- Safety layer catches the error
- You see: `"Safety check failed: Grouping variable 'INVALID_COLUMN' not found in dataset..."`
- Analysis does NOT execute

---

#### ✅ Test 3: Valid Case-Control (Should Pass)
**Query:** `"Compare TP53_Mutation frequency in Stage III vs Stage I"`

**Expected:**
- Safety layer passes
- Analysis executes
- You see odds ratio and p-value

---

#### ❌ Test 4: Missing Target Variable (Should Fail)
**Query:** `"Compare frequency in Stage III vs Stage I"`

**Expected:**
- Safety layer catches missing target variable
- Error message about required target variable
- Analysis does NOT execute

---

#### ✅ Test 5: Valid Association Scan (Should Pass)
**Query:** `"Tell me everything associated with TUMOR_STAGE"`

**Expected:**
- Safety layer passes
- Analysis executes
- You see association results

---

#### ❌ Test 6: Invalid Target for Association Scan (Should Fail)
**Query:** `"Tell me everything associated with FAKE_COLUMN"`

**Expected:**
- Safety layer catches invalid column
- Error: `"Target variable 'FAKE_COLUMN' not found in dataset..."`
- Analysis does NOT execute

---

#### ❌ Test 7: Invalid Analysis Mode (Should Fail)
If you manually modify the LLM output to have an invalid mode, it should catch it.

---

## Manual Testing (Advanced)

### Test the Safety Layer Directly in Python

Create a test script `test_safety_layer.py`:

```python
import sys
import os
sys.path.insert(0, 'src')

from safety_layer import load_dataset_schema, validate_analysis_plan, ValidationError

# Test 1: Load schema
dataset_path = "data/TCGA_COAD"
schema = load_dataset_schema(dataset_path)
print(f"Loaded schema for: {schema.name}")
print(f"Variables: {list(schema.variables.keys())}")

# Test 2: Valid plan
valid_plan = {
    "mode": "survival",
    "dataset_path": dataset_path,
    "group_by": "TP53_Mutation",
    "time_col": "OS_MONTHS",
    "event_col": "OS_STATUS"
}
try:
    validate_analysis_plan(valid_plan, schema)
    print("✅ Valid plan passed!")
except ValidationError as e:
    print(f"❌ Valid plan failed: {e}")

# Test 3: Invalid column
invalid_plan = {
    "mode": "survival",
    "dataset_path": dataset_path,
    "group_by": "FAKE_COLUMN",
    "time_col": "OS_MONTHS",
    "event_col": "OS_STATUS"
}
try:
    validate_analysis_plan(invalid_plan, schema)
    print("❌ Invalid plan should have failed!")
except ValidationError as e:
    print(f"✅ Invalid plan correctly rejected: {e}")

# Test 4: Missing required field
missing_field_plan = {
    "mode": "survival",
    "dataset_path": dataset_path,
    # Missing group_by
}
try:
    validate_analysis_plan(missing_field_plan, schema)
    print("❌ Plan with missing field should have failed!")
except ValidationError as e:
    print(f"✅ Missing field correctly caught: {e}")
```

Run it:
```bash
python test_safety_layer.py
```

---

## What to Look For

### Success Indicators:
1. ✅ Valid queries execute normally (no change in behavior)
2. ✅ Invalid queries show clear error messages BEFORE analysis runs
3. ✅ Error messages are user-friendly and show available columns
4. ✅ No crashes or exceptions in the console

### Debugging:
- Check the browser console (F12) for any JavaScript errors
- Check the terminal where Streamlit is running for Python errors
- Expand "See AI Logic" in the app to see what the LLM generated
- If validation fails silently, check that the error handling is working

---

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'safety_layer'"
**Solution:** Make sure you're running from the project root, and `src/` is in the Python path.

### Issue: "FileNotFoundError: data_table.tsv not found"
**Solution:** The safety layer will fall back to `main_data.tsv`. This is expected and fine.

### Issue: Ollama connection errors
**Solution:** Make sure Ollama is running: `ollama serve` (or it should auto-start)

### Issue: Safety layer not catching errors
**Solution:** Check that the validation is being called (add print statements or check the code flow)

