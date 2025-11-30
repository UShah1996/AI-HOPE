import ollama
import json
import re


class LLMAgent:
    def __init__(self, model_name="llama3"):
        self.model = model_name

    def _clean_json(self, content):
        """Helper to extract JSON from LLM chatter."""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            content = match.group(1)
        return content

    def check_clarification_needed(self, query, column_names):
        """
        [Agent 1: Input Validator]
        Checks if the query is specific enough.
        """
        prompt = f"""
        You are a strict data librarian. 
        User Query: "{query}"
        Available Columns: {column_names}

        Task: Determine if the query is clear enough to run a statistical test (Survival, Case-Control, or Scan).
        - If it refers to columns that likely exist or implies a clear comparison, output "CLEAR".
        - If it is too vague (e.g., "Analyze data", "Is it bad?", "Check survival" without saying by what), output a clarifying question.

        Output ONLY "CLEAR" or the question.
        """
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content'].strip()

            if "CLEAR" in result.upper():
                return None  # No clarification needed
            return result  # Return the question (e.g., "Which variable do you want to compare?")
        except:
            return None  # Fail safe to proceed if LLM errors

    def interpret_query(self, user_query, column_names):
        """
        [Agent 2: Planner]
        Generates the initial JSON logic.
        """
        system_prompt = f"""
        You are AI-HOPE. Convert the query into JSON logic.
        Available Data Attributes: {column_names}

        RULES:
        1. Identify 'target_variable' and 'cohort' definitions.
        2. Use ONLY operators: "is", "is not", "greater than", "less than", "is in".

        EXAMPLE INPUT: "Compare TP53 mutations in early vs late stage"
        EXAMPLE JSON OUTPUT:
        {{
            "analysis_type": "case_control",
            "target_variable": "TP53_Mutation",
            "case_condition": "TUMOR_STAGE is in {{'Stage III', 'Stage IV'}}",
            "control_condition": "TUMOR_STAGE is in {{'Stage I', 'Stage II'}}"
        }}

        Return ONLY valid JSON.
        """

        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ])

        json_text = self._clean_json(response['message']['content'])

        # --- [Agent 3: Verifier Handoff] ---
        # Instead of returning immediately, we pass the logic to the Verifier
        return self.verify_logic(json_text, column_names)

    def verify_logic(self, json_str, column_names):
        """
        [Agent 3: Verifier]
        Sanity checks the generated JSON against actual columns to prevent hallucinations.
        """
        verifier_prompt = f"""
        You are a Code Reviewer.
        Original JSON Logic: {json_str}
        Allowed Columns: {column_names}

        Task: 
        1. Check if 'target_variable' and columns in conditions actually exist in Allowed Columns.
        2. If a column is misspelled or hallucinated (e.g. 'KRAS_Status' instead of 'KRAS_mutation_status'), CORRECT it to the exact match from Allowed Columns.
        3. Return the CORRECTED JSON string only.
        """

        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': verifier_prompt}])
            corrected_json = self._clean_json(response['message']['content'])
            return json.loads(corrected_json)
        except Exception as e:
            # If verifier fails, try to load the original JSON as a fallback
            try:
                return json.loads(json_str)
            except:
                return {"error": f"Parsing failed: {str(e)}"}

    def suggest_analysis(self, query):
        """
        Determines analysis category.
        """
        prompt = f"""
        Classify this clinical research question into exactly one of these three categories:
        1. 'Survival Analysis': Questions about survival time, outcomes, Kaplan-Meier, or hazard ratios.
        2. 'Case-Control': Questions comparing TWO specific groups (e.g., "Stage I vs Stage IV", "Mutated vs Wild-type").
        3. 'Association Scan': Open-ended discovery questions (e.g., "What is associated with X?", "Find all correlations").

        USER QUESTION: "{query}"
        OUTPUT (Return ONLY the category name):
        """
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content'].strip().strip('"').strip("'")
        except:
            return "Error"