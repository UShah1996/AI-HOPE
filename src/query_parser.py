import re
import pandas as pd

def parse_filter(expression: str):
    # Converts natural expressions to Pythonic conditions
    expr = expression.replace("is greater than", ">").replace("is less than", "<")
    expr = expr.replace("is", "==").replace("equals", "==")
    return expr

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, condition in filters.items():
        expr = f"df['{col}']{condition}"
        df = df.query(expr)
    return df
