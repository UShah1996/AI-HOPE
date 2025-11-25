import re
import pandas as pd


class QueryParser:
    """
    Parses natural language-like logic returned by the LLM into executable pandas queries.
    """

    def __init__(self):
        # Maps paper-defined natural language operators to Pandas syntax
        self.operator_map = {
            "is": "==",
            "is not": "!=",
            "greater than": ">",
            "less than": "<",
            "is in": "in",
            "is not in": "not in"
        }

    def parse_statement(self, llm_output_string):
        """
        Converts a structured string like "Age > 30" into a DataFrame filter mask.
        """
        if not llm_output_string:
            return None, None, None

        # 1. Handle Set Inclusion: "Stage is in {Stage I, Stage II}"
        set_match = re.search(r"(.+?)\s+(is in|is not in)\s+\{(.+?)\}", llm_output_string, re.IGNORECASE)

        if set_match:
            col, op, val_str = set_match.groups()
            values = [v.strip().strip("'").strip('"') for v in val_str.split(',')]
            return col.strip(), self.operator_map[op.lower()], values

        # 2. Handle Basic Comparisons
        simple_match = re.search(r"(.+?)\s+(is|is not|greater than|less than|==|!=|>|<)\s+(.+)", llm_output_string,
                                 re.IGNORECASE)

        if simple_match:
            col, op, val = simple_match.groups()
            val = val.strip()
            # Try converting to float if it looks like a number
            try:
                val = float(val)
            except ValueError:
                pass  # Keep as string

            pandas_op = self.operator_map.get(op.lower(), op)
            return col.strip(), pandas_op, val

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