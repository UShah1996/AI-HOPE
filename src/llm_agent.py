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
        1. Identify the 'target' variable (e.g., Survival, Mutation Status).
        2. Identify the 'cohort' definitions (e.g., Late Stage vs Early Stage).
        3. Use ONLY these operators: "is", "is not", "greater than", "less than", "is in".

        EXAMPLE INPUT: "Compare TP53 mutations in early vs late stage CRC"
        EXAMPLE JSON OUTPUT:
        {{
            "analysis_type": "case_control",
            "target_variable": "TP53_Mutation",
            "case_condition": "TUMOR_STAGE is in {{'Stage III', 'Stage IV'}}",
            "control_condition": "TUMOR_STAGE is in {{'Stage I', 'Stage II'}}"
        }}

        Return ONLY valid JSON. Do not add conversational text.
        """

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_query}
            ])

            content = response['message']['content']

            # --- IMPROVED CLEANING LOGIC ---
            # 1. Try to find JSON inside markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()

            # 2. If that fails, look for the first '{' and last '}'
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)
            # -------------------------------

            return json.loads(content)

        except Exception as e:
            # Print the raw content to the terminal for debugging
            print(f"DEBUG: Failed LLM Content: {content if 'content' in locals() else 'No content'}")
            return {"error": f"LLM Parsing Failed: {str(e)}"}

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