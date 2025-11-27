"""
Quick test script for the safety layer.
Run this to verify the safety layer works independently.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from safety_layer import load_dataset_schema, validate_analysis_plan, ValidationError

def test_safety_layer():
    print("=" * 60)
    print("Testing Safety Layer")
    print("=" * 60)
    
    dataset_path = "data/TCGA_COAD"
    
    # Test 1: Load schema
    print("\n[Test 1] Loading dataset schema...")
    try:
        schema = load_dataset_schema(dataset_path)
        print(f"[OK] Loaded schema for: {schema.name}")
        print(f"   Found {len(schema.variables)} variables:")
        for var_name, var_schema in list(schema.variables.items())[:5]:
            print(f"     - {var_name} ({var_schema.var_type})")
        if len(schema.variables) > 5:
            print(f"     ... and {len(schema.variables) - 5} more")
    except Exception as e:
        print(f"[FAIL] Failed to load schema: {e}")
        return
    
    # Test 2: Valid survival plan
    print("\n[Test 2] Testing valid survival analysis plan...")
    valid_plan = {
        "mode": "survival",
        "dataset_path": dataset_path,
        "group_by": "TP53_Mutation",
        "time_col": "OS_MONTHS",
        "event_col": "OS_STATUS"
    }
    try:
        validate_analysis_plan(valid_plan, schema)
        print("[OK] Valid plan passed validation!")
    except ValidationError as e:
        print(f"[FAIL] Valid plan failed: {e}")
    
    # Test 3: Invalid column name
    print("\n[Test 3] Testing plan with invalid column name...")
    invalid_plan = {
        "mode": "survival",
        "dataset_path": dataset_path,
        "group_by": "FAKE_COLUMN_THAT_DOES_NOT_EXIST",
        "time_col": "OS_MONTHS",
        "event_col": "OS_STATUS"
    }
    try:
        validate_analysis_plan(invalid_plan, schema)
        print("[FAIL] Invalid plan should have failed but didn't!")
    except ValidationError as e:
        print(f"[OK] Invalid plan correctly rejected: {e}")
    
    # Test 4: Missing required field
    print("\n[Test 4] Testing plan with missing required field...")
    missing_field_plan = {
        "mode": "survival",
        "dataset_path": dataset_path,
        # Missing group_by
    }
    try:
        validate_analysis_plan(missing_field_plan, schema)
        print("[FAIL] Plan with missing field should have failed but didn't!")
    except ValidationError as e:
        print(f"[OK] Missing field correctly caught: {e}")
    
    # Test 5: Valid case-control plan
    print("\n[Test 5] Testing valid case-control plan...")
    case_control_plan = {
        "mode": "case_control",
        "dataset_path": dataset_path,
        "target": "TP53_Mutation"
    }
    try:
        validate_analysis_plan(case_control_plan, schema)
        print("[OK] Valid case-control plan passed!")
    except ValidationError as e:
        print(f"[FAIL] Valid case-control plan failed: {e}")
    
    # Test 6: Invalid target for case-control
    print("\n[Test 6] Testing case-control with invalid target...")
    invalid_target_plan = {
        "mode": "case_control",
        "dataset_path": dataset_path,
        "target": "INVALID_TARGET"
    }
    try:
        validate_analysis_plan(invalid_target_plan, schema)
        print("[FAIL] Invalid target should have failed but didn't!")
    except ValidationError as e:
        print(f"[OK] Invalid target correctly rejected: {e}")
    
    # Test 7: Valid association scan
    print("\n[Test 7] Testing valid association scan plan...")
    scan_plan = {
        "mode": "association_scan",
        "dataset_path": dataset_path,
        "target": "TUMOR_STAGE"
    }
    try:
        validate_analysis_plan(scan_plan, schema)
        print("[OK] Valid association scan plan passed!")
    except ValidationError as e:
        print(f"[FAIL] Valid association scan plan failed: {e}")
    
    # Test 8: Invalid mode
    print("\n[Test 8] Testing plan with invalid mode...")
    invalid_mode_plan = {
        "mode": "invalid_mode_xyz",
        "dataset_path": dataset_path,
    }
    try:
        validate_analysis_plan(invalid_mode_plan, schema)
        print("[FAIL] Invalid mode should have failed but didn't!")
    except ValidationError as e:
        print(f"[OK] Invalid mode correctly rejected: {e}")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_safety_layer()

