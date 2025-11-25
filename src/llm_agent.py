# import json
# from intents import AnalysisIntent
#
# class AIHOPEAgent:
#     def __init__(self, model_name="google/flan-t5-base"):
#         from transformers import pipeline
#         self.generator = pipeline("text2text-generation", model=model_name)
#
#     def interpret_query(self, user_input: str) -> AnalysisIntent:
#         prompt = f"""
#         You are AI-HOPE, an assistant that converts biomedical research questions
#         into structured analysis instructions.
#
#         Example output JSON:
#         {{
#           "query_type": "survival",
#           "dataset": "TCGA_COAD",
#           "target_variable": "KRAS_mutation_status",
#           "group_variable": "FOLFOX_treatment",
#           "filters": {{"cancer_type": "colorectal"}}
#         }}
#
#         Now respond ONLY with valid JSON for this question:
#         "{user_input}"
#         """
#
#         response = self.generator(prompt, max_new_tokens=256)[0]["generated_text"]
#         print("\n🧩 Raw model output:\n", response, "\n")
#
#         # Try to find JSON in the response
#         try:
#             start = response.index("{")
#             end = response.rindex("}") + 1
#             json_text = response[start:end]
#             parsed = json.loads(json_text)
#         except Exception:
#             # fallback — if model returned plain text or malformed JSON
#             print("⚠️ Could not parse model output as JSON, using fallback intent.")
#             parsed = {
#                 "query_type": "survival",
#                 "dataset": "TCGA_COAD",
#                 "target_variable": "KRAS_mutation_status",
#                 "group_variable": "FOLFOX_treatment"
#             }
#
#         # Fill missing optional fields
#         for field in ["dataset", "target_variable", "group_variable"]:
#             if field not in parsed or not parsed[field]:
#                 parsed[field] = "unknown"
#
#         return AnalysisIntent(**parsed)
from intents import AnalysisIntent
from difflib import get_close_matches
import json


def validate_columns(self, parsed_json, available_columns):
    """
    Sanity checks the LLM output against the actual dataset index.
    """
    # Extract all columns mentioned in the JSON logic
    mentioned_cols = []
    if "target_variable" in parsed_json: mentioned_cols.append(parsed_json["target_variable"])

    # Check if they exist
    for col in mentioned_cols:
        if col not in available_columns:
            # Try to auto-correct using string similarity
            suggestion = get_close_matches(col, available_columns, n=1)
            if suggestion:
                parsed_json["target_variable"] = suggestion[0]  # Auto-fix
            else:
                return False, f"Error: Column '{col}' not found in dataset."

    return True, parsed_json

class AIHOPEAgent:
    def __init__(self, model_name=None):
        self.generator = None

    def interpret_query(self, user_input: str):
        print("Mock LLM: parsing simulated intent.")
        return AnalysisIntent(
            query_type="survival",
            dataset="TCGA_COAD",
            target_variable="KRAS_mutation_status",
            group_variable="FOLFOX_treatment"
        )
