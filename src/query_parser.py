import re
import pandas as pd


class QueryParser:
    """
    Parses natural language-like logic returned by the LLM into executable pandas queries.
    """

    def __init__(self, df=None):
        # Maps paper-defined natural language operators to Pandas syntax
        self.operator_map = {
            "is": "==",
            "is not": "!=",
            "greater than": ">",
            "less than": "<",
            "is in": "in",
            "is not in": "not in"
        }
        self.df = df
        # Semantic mappings for common clinical terms
        self.semantic_mappings = {
            "late-stage": ["Stage III", "Stage IV"],
            "late stage": ["Stage III", "Stage IV"],
            "advanced stage": ["Stage III", "Stage IV"],
            "early-stage": ["Stage I", "Stage II"],
            "early stage": ["Stage I", "Stage II"],
            "early": ["Stage I", "Stage II"],
            "late": ["Stage III", "Stage IV"]
        }

    def _map_semantic_value(self, col, val):
        """
        Maps semantic terms (e.g., 'late-stage') to actual values in the data.
        Returns the mapped value or original value if no mapping found.
        """
        if not isinstance(val, str) or self.df is None or col not in self.df.columns:
            return val
        
        val_lower = val.lower().strip()
        
        # Check semantic mappings first
        if val_lower in self.semantic_mappings:
            mapped_values = self.semantic_mappings[val_lower]
            # Check if any mapped values exist in the actual data
            actual_values = self.df[col].unique()
            matching_values = [v for v in mapped_values if v in actual_values]
            if matching_values:
                # If multiple matches, return as list for "in" operation
                if len(matching_values) > 1:
                    return matching_values
                return matching_values[0]
        
        # If no semantic mapping, try to find close matches in actual data
        actual_values = [str(v).lower() for v in self.df[col].unique()]
        if val_lower in actual_values:
            # Find the actual case-sensitive value
            for actual_val in self.df[col].unique():
                if str(actual_val).lower() == val_lower:
                    return actual_val
        
        return val

    def parse_statement(self, llm_output_string):
        """
        Converts a structured string like "Age > 30" into a DataFrame filter mask.
        Handles OR conditions by converting them to "is in" operations.
        """
        if not llm_output_string:
            return None, None, None

        # 0. Handle OR conditions: "COL is 'A' or COL is 'B'" -> convert to "COL is in ['A', 'B']"
        # Check if the string contains "or" (case insensitive)
        if " or " in llm_output_string.lower():
            # Try to match pattern: COLUMN is 'VALUE' or COLUMN is 'VALUE'
            # First, try to extract the column name from the first condition
            first_condition_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s+is\s+['\"]([^'\"]+)['\"]", llm_output_string, re.IGNORECASE)
            if first_condition_match:
                col = first_condition_match.group(1)
                first_value = first_condition_match.group(2)
                
                # Check if all OR conditions use the same column
                # Split by " or " and check each part
                parts = re.split(r"\s+or\s+", llm_output_string, flags=re.IGNORECASE)
                values = []
                all_same_col = True
                
                for part in parts:
                    part_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s+is\s+['\"]([^'\"]+)['\"]", part.strip(), re.IGNORECASE)
                    if part_match:
                        part_col = part_match.group(1)
                        part_val = part_match.group(2)
                        if part_col.lower() == col.lower():
                            values.append(part_val)
                        else:
                            all_same_col = False
                            break
                    else:
                        all_same_col = False
                        break
                
                # If all OR conditions use the same column, convert to "is in"
                if all_same_col and len(values) > 1:
                    col = col.strip()
                    # Map semantic values if dataframe is available
                    if self.df is not None and col in self.df.columns:
                        mapped_values = []
                        for v in values:
                            mapped = self._map_semantic_value(col, v)
                            if isinstance(mapped, list):
                                mapped_values.extend(mapped)
                            else:
                                mapped_values.append(mapped)
                        values = list(set(mapped_values))  # Remove duplicates
                    else:
                        values = [v.strip() for v in values]
                    
                    return col, "in", values

        # 1. Handle Set Inclusion: "Stage is in {Stage I, Stage II}"
        set_match = re.search(r"(.+?)\s+(is in|is not in)\s+\{(.+?)\}", llm_output_string, re.IGNORECASE)

        if set_match:
            col, op, val_str = set_match.groups()
            values = [v.strip().strip("'").strip('"') for v in val_str.split(',')]
            col = col.strip()
            # Map semantic values if dataframe is available
            if self.df is not None and col in self.df.columns:
                mapped_values = []
                for v in values:
                    mapped = self._map_semantic_value(col, v)
                    if isinstance(mapped, list):
                        mapped_values.extend(mapped)
                    else:
                        mapped_values.append(mapped)
                values = mapped_values
            return col, self.operator_map[op.lower()], values

        # 2. Handle Basic Comparisons
        simple_match = re.search(r"(.+?)\s+(is|is not|greater than|less than|==|!=|>|<)\s+(.+)", llm_output_string,
                                 re.IGNORECASE)

        if simple_match:
            col, op, val = simple_match.groups()
            val = val.strip()
            # Strip quotes if present (handles 'Stage IV' or "Stage IV")
            if isinstance(val, str) and ((val.startswith("'") and val.endswith("'")) or 
                                         (val.startswith('"') and val.endswith('"'))):
                val = val[1:-1]
            
            col = col.strip()
            
            # Map semantic values before type conversion
            if self.df is not None and col in self.df.columns:
                mapped_val = self._map_semantic_value(col, val)
                # If mapping returned a list, we need to use "in" operator
                if isinstance(mapped_val, list):
                    return col, "in", mapped_val
                val = mapped_val
            
            # Try converting to float if it looks like a number
            try:
                val = float(val)
            except ValueError:
                pass  # Keep as string

            pandas_op = self.operator_map.get(op.lower(), op)
            return col, pandas_op, val

        return None, None, None

    def apply_filter(self, df, col, op, val):
        """
        Executes the parsed logic on the actual Dataframe.
        """
        try:
            if not col or not op:
                return df

            if op == "in":
                return df[df[col].isin(val)]
            elif op == "not in":
                return df[~df[col].isin(val)]
            else:
                if isinstance(val, str):
                    query = f"`{col}` {op} '{val}'"
                else:
                    query = f"`{col}` {op} {val}"
                return df.query(query)
        except Exception as e:
            # Return original DF if filter fails so app doesn't crash
            return df