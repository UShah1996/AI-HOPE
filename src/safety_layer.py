from dataclasses import dataclass
from typing import Dict, Literal, Any, List, TypedDict
import pandas as pd
import os


VariableType = Literal["categorical", "continuous", "time", "event"]


@dataclass
class VariableSchema:
    name: str
    var_type: VariableType


@dataclass
class DatasetSchema:
    name: str
    variables: Dict[str, VariableSchema]


def load_dataset_schema(dataset_path: str) -> DatasetSchema:
    """
    Loads the dataset schema by reading data_table.tsv (or main_data.tsv as fallback).
    Infers variable types from the data.
    """
    # Try data_table.tsv first (as per user instructions), then fallback to main_data.tsv
    data_table_path = os.path.join(dataset_path, "data_table.tsv")
    if not os.path.exists(data_table_path):
        data_table_path = os.path.join(dataset_path, "main_data.tsv")
    
    if not os.path.exists(data_table_path):
        raise FileNotFoundError(f"Neither data_table.tsv nor main_data.tsv found in {dataset_path}")
    
    df = pd.read_csv(data_table_path, sep="\t")
    variables: Dict[str, VariableSchema] = {}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            vtype: VariableType = "continuous"
        else:
            vtype = "categorical"
        variables[col] = VariableSchema(name=col, var_type=vtype)
    
    # Simple heuristic refinements
    for col in df.columns:
        cl = col.lower()
        if "month" in cl or "time" in cl:
            variables[col].var_type = "time"
        if "status" in cl or "event" in cl:
            variables[col].var_type = "event"
    
    return DatasetSchema(
        name=os.path.basename(dataset_path.rstrip("/")),
        variables=variables,
    )


class FilterSpec(TypedDict):
    column: str
    op: str      # "==", "!=", ">", "<", ">=", "<=", "in", "not in", "between"
    value: Any


class AnalysisPlan(TypedDict, total=False):
    mode: str             # e.g. "survival", "case_control", "association_scan"
    dataset_path: str
    filters: List[FilterSpec]
    group_by: str         # grouping column
    time_col: str         # for survival
    event_col: str        # for survival
    test: str             # optional
    target: str           # optional, e.g. for global scan
    # Additional fields to match existing logic_json structure
    target_variable: str  # for case_control and association_scan
    grouping_variable: str  # for survival
    case_condition: str   # for case_control
    control_condition: str  # for case_control


class ValidationError(Exception):
    """Raised when the safety layer rejects an analysis plan."""
    pass


def validate_analysis_plan(plan: AnalysisPlan, schema: DatasetSchema) -> None:
    """
    Raise ValidationError with a clear message if:
    - mode is unsupported
    - referenced columns do not exist
    - required fields are missing for that mode
    - basic type checks fail (e.g., time_col not numeric/time)
    """
    mode = plan.get("mode", "").lower()
    
    # Check if mode is supported
    supported_modes = ["survival", "case_control", "case-control", "association_scan", "association scan", "global_scan", "global scan"]
    if not mode or not any(supported in mode for supported in supported_modes):
        raise ValidationError(f"Unsupported analysis mode: '{mode}'. Supported modes: survival, case_control, association_scan")
    
    # Get all column names from schema
    available_columns = set(schema.variables.keys())
    
    # Validate survival mode
    if "survival" in mode:
        group_by = plan.get("group_by") or plan.get("grouping_variable")
        if not group_by:
            raise ValidationError("Survival analysis requires a grouping variable (group_by or grouping_variable)")
        
        if group_by not in available_columns:
            raise ValidationError(f"Grouping variable '{group_by}' not found in dataset. Available columns: {', '.join(sorted(available_columns))}")
        
        # Check time_col and event_col (optional, with defaults)
        time_col = plan.get("time_col", "OS_MONTHS")
        event_col = plan.get("event_col", "OS_STATUS")
        
        if time_col not in available_columns:
            raise ValidationError(f"Time column '{time_col}' not found in dataset. Available columns: {', '.join(sorted(available_columns))}")
        
        if event_col not in available_columns:
            raise ValidationError(f"Event column '{event_col}' not found in dataset. Available columns: {', '.join(sorted(available_columns))}")
        
        # Type check: time_col should be continuous or time type
        if time_col in schema.variables:
            time_var_type = schema.variables[time_col].var_type
            if time_var_type not in ["continuous", "time"]:
                raise ValidationError(f"Time column '{time_col}' should be numeric/time type, but detected as '{time_var_type}'")
    
    # Validate case_control mode
    elif "case" in mode or "control" in mode:
        target = plan.get("target") or plan.get("target_variable")
        if not target:
            raise ValidationError("Case-control analysis requires a target variable (target or target_variable)")
        
        if target not in available_columns:
            raise ValidationError(f"Target variable '{target}' not found in dataset. Available columns: {', '.join(sorted(available_columns))}")
        
        # Note: case_condition and control_condition are parsed strings, not direct column references
        # We validate them exist in the parser, but here we just check target exists
    
    # Validate association_scan/global_scan mode
    elif "scan" in mode or "association" in mode:
        target = plan.get("target") or plan.get("target_variable")
        if not target:
            raise ValidationError("Association scan requires a target variable (target or target_variable)")
        
        if target not in available_columns:
            raise ValidationError(f"Target variable '{target}' not found in dataset. Available columns: {', '.join(sorted(available_columns))}")
    
    # Validate filters if present
    filters = plan.get("filters", [])
    for filter_spec in filters:
        col = filter_spec.get("column")
        if col and col not in available_columns:
            raise ValidationError(f"Filter column '{col}' not found in dataset. Available columns: {', '.join(sorted(available_columns))}")

