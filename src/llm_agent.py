import ollama
import json
import re


class LLMAgent:
    """
    The central intelligence of AI-HOPE.
    """

    def __init__(self, model_name="llama3"):
        self.model = model_name

    def interpret_query(self, user_query, column_names):
        """
        Uses Llama3 to parse natural language into structured logic rules.
        """
        system_prompt = f"""
        You are AI-HOPE, a clinical research assistant.
        Your task is to convert a natural language query into a STRUCTURED JSON format.

        Available Data Attributes: {column_names}

        RULES:
        1. Identify the 'target' variable (e.g., Survival, Mutation Status, Tumor Stage).
        2. Identify the 'cohort' definitions (e.g., Late Stage vs Early Stage) if applicable.
        3. Use ONLY these operators: "is", "is not", "greater than", "less than", "is in".
        4. For association scan queries, identify the target variable that other variables should be tested against.

        EXAMPLE 1 - Case Control:
        INPUT: "Compare TP53 mutations in early vs late stage CRC"
        OUTPUT:
        {{
            "analysis_type": "case_control",
            "target_variable": "TP53_Mutation",
            "case_condition": "TUMOR_STAGE is in {{'Stage III', 'Stage IV'}}",
            "control_condition": "TUMOR_STAGE is in {{'Stage I', 'Stage II'}}"
        }}

        EXAMPLE 2 - Association Scan:
        INPUT: "What variables are linked to TUMOR_STAGE?"
        OUTPUT:
        {{
            "analysis_type": "association_scan",
            "target_variable": "TUMOR_STAGE"
        }}

        EXAMPLE 3 - Survival Analysis:
        INPUT: "Compare survival between TP53 mutated and wild-type patients"
        OUTPUT:
        {{
            "analysis_type": "survival",
            "grouping_variable": "TP53_Mutation"
        }}

        Return ONLY valid JSON. Do not add conversational text, explanations, or markdown formatting.
        """

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_query}
            ])

            content = response['message']['content'].strip()

            # --- IMPROVED CLEANING LOGIC ---
            # 1. Try to find JSON inside markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # Handle generic code blocks
                parts = content.split("```")
                if len(parts) > 1:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()

            # 2. If that fails, look for the first '{' and last '}'
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)
            
            # 3. Remove any leading/trailing whitespace or newlines
            content = content.strip()
            # -------------------------------

            parsed_json = json.loads(content)
            return parsed_json

        except json.JSONDecodeError as e:
            # Include the actual content in the error for debugging
            error_msg = f"LLM Parsing Failed: {str(e)}"
            if 'content' in locals():
                error_msg += f"\nRaw LLM Response: {content[:200]}..."  # First 200 chars
            return {"error": error_msg, "raw_content": content if 'content' in locals() else "No content received"}
        except Exception as e:
            # Handle other exceptions (e.g., Ollama connection issues)
            error_msg = f"LLM Error: {str(e)}"
            if 'content' in locals():
                error_msg += f"\nRaw LLM Response: {content[:200]}..."
            return {"error": error_msg, "raw_content": content if 'content' in locals() else "No content received"}

    def suggest_analysis(self, query):
        """
        Determines if the request is a Survival Analysis, Association Scan, or Case-Control.
        """
        prompt = f"Classify this clinical question into one category: 'Survival Analysis', 'Case-Control', or 'Association Scan'. Question: '{query}'. Return ONLY the category name."

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content'].strip().strip('"').strip("'")
        except Exception as e:
            return "Error"