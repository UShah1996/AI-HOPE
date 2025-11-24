from typing import Optional, Dict
from pydantic import BaseModel

class AnalysisIntent(BaseModel):
    query_type: str
    dataset: Optional[str] = None
    target_variable: Optional[str] = None
    group_variable: Optional[str] = None
    filters: Optional[Dict[str, str]] = None
    notes: Optional[str] = None
