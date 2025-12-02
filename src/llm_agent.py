import ollama
import json
import re
from difflib import get_close_matches


class LLMAgent:
    def __init__(self, model_name="llama3"):
        self.model = model_name

    def _clean_json(self, content):
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            content = match.group(1)
        return content

    def check_clarification_needed(self, query, column_names):
        """[Agent 1: Input Validator]"""
        prompt = f"""
        You are a strict data librarian. 
        User Query: "{query}"
        Available Columns: {column_names}

        Task: Determine if the query is clear enough to run a statistical test.
        GUIDELINES:
        - "Compare survival outcomes..." -> CLEAR.
        - "Does [Variable] affect survival?" -> CLEAR.
        - "Compare [Group A] vs [Group B]" -> CLEAR.
        - "Check KRAS status" -> CLEAR (Implies Prevalence).
        - "Analyze data" -> NOT CLEAR.
        - "Is it good?" -> NOT CLEAR.

        Output ONLY "CLEAR" or the clarifying question.
        """
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content'].strip()
            if "CLEAR" in result.upper(): return None
            return result
        except:
            return None

    def interpret_query(self, user_query, column_names):
        """[Agent 2: Planner]"""
        system_prompt = f"""
        You are AI-HOPE. Convert the query into JSON logic.
        Available Data Attributes: {column_names}

        RULES:
        1. Output MUST use one of these 'analysis_type' values: 'case_control', 'survival', 'scan'.
        2. DO NOT use 'descriptive', 'summary', or 'prevalence'. 
           - If user asks for simple counts (e.g. "Check KRAS"), use 'scan' or 'case_control' with 'All' as cohort.
        3. Identify 'target_variable' and 'cohort' definitions.

        EXAMPLE 1 (Comparison): "Compare TP53 in Stage IV vs I"
        {{ "analysis_type": "case_control", "target_variable": "TP53_Mutation", "case_condition": "TUMOR_STAGE is 'Stage IV'", "control_condition": "TUMOR_STAGE is 'Stage I'" }}

        EXAMPLE 2 (Prevalence): "Check KRAS status"
        {{ "analysis_type": "scan", "target_variable": "KRAS_mutation_status" }}

        Return ONLY valid JSON.
        """

        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ])

        json_text = self._clean_json(response['message']['content'])
        return self.verify_logic(json_text, column_names)

    def verify_logic(self, json_str, column_names):
        """[Agent 3: Verifier]"""
        try:
            logic = json.loads(json_str)

            def fix_col(val):
                if val and isinstance(val, str) and val not in column_names:
                    matches = get_close_matches(val, column_names, n=1, cutoff=0.6)
                    if matches: return matches[0]
                return val

            if 'target_variable' in logic: logic['target_variable'] = fix_col(logic['target_variable'])
            if 'group_by' in logic: logic['group_by'] = fix_col(logic['group_by'])
            return logic
        except:
            return {"error": "Parsing failed"}

    def suggest_analysis(self, query):
        prompt = f"""
        Classify this clinical research question into exactly one of these three categories:
        1. 'Survival Analysis': Questions about survival time, outcomes, Kaplan-Meier.
        2. 'Case-Control': Questions comparing groups (e.g., "Stage I vs IV") or Prevalence (e.g. "Count KRAS").
        3. 'Association Scan': Open-ended discovery questions.

        USER QUESTION: "{query}"
        OUTPUT (Return ONLY the category name):
        """
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content'].strip().strip('"').strip("'")
        except:
            return "Error"