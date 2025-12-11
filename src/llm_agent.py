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
        elif "```" in content:
            # Handle other code blocks
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are code blocks
                    if "{" in part and "}" in part:
                        content = part.strip()
                        break
        
        # Try to extract JSON object
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            content = match.group(1)
        
        # Remove JSON comments (both // and /* */ style)
        # Remove single-line comments (// ...)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove any trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        return content

    def check_clarification_needed(self, query, column_names):
        """[Agent 1: Input Validator]"""
        # Quick pattern check for obviously vague queries before LLM call
        query_lower = query.lower().strip()
        vague_patterns = [
            "is the data good", "is it good", "is data good",
            "how is the data", "how's the data", "how good is",
            "analyze data", "analyze the data", "analyze this",
            "tell me about the data", "what about the data",
            "is this good", "is this data good", "data quality",
            "check data", "review data", "examine data"
        ]
        
        for pattern in vague_patterns:
            if pattern in query_lower:
                return "This query is too vague. Please specify: What specific analysis do you want to perform? (e.g., survival analysis, comparison between groups, association scan)"
        
        prompt = f"""
        You are a strict data librarian. 
        User Query: "{query}"
        Available Columns: {column_names}

        Task: Determine if the query is clear enough to run a statistical test.
        GUIDELINES:
        - "Compare survival outcomes..." -> CLEAR.
        - "Compare survival for [Variable]" -> CLEAR (Survival analysis, variable name will be verified later).
        - "Does [Variable] affect survival?" -> CLEAR.
        - "Perform survival analysis..." -> CLEAR.
        - "Compare [Variable] frequency in [Group A] vs [Group B]" -> CLEAR (Case-control comparison).
        - "Compare [Variable] in [Group A] vs [Group B]" -> CLEAR (Case-control comparison).
        - "Compare [Group A] vs [Group B]" -> CLEAR.
        - "Check KRAS status" -> CLEAR (Implies Prevalence).
        - "Tell me everything associated with [Variable]" -> CLEAR (Association Scan).
        - "Find variables correlated with [Variable]" -> CLEAR (Association Scan).
        - "Run a global association scan..." -> CLEAR (Association Scan).
        - "What is associated with [Variable]?" -> CLEAR (Association Scan).
        - "Analyze data" -> NOT CLEAR (too vague).
        - "Is the data good?" -> NOT CLEAR (too vague, no specific analysis requested).
        - "Is it good?" -> NOT CLEAR (too vague).
        - "Tell me about the data" -> NOT CLEAR (too vague).
        - Any query asking about data quality without specifying analysis -> NOT CLEAR.

        IMPORTANT: 
        1. If the query specifies an analysis type (survival, comparison, scan) and mentions variables/groups, 
           it is CLEAR even if the variable names might be wrong or don't exist. The Verifier will handle validation later.
        2. The Clarifier's job is ONLY to check if the query is specific enough, NOT to validate if variables exist.
        3. Examples of CLEAR queries even if variables don't exist:
           - "Compare BRAF_mutation frequency in male vs female" -> CLEAR (specific comparison requested)
           - "Compare survival for KRAS_Status" -> CLEAR (specific analysis type and variable)
        
        If the query is vague or doesn't specify what analysis to run, return a clarifying question.
        If the query clearly specifies an analysis type and variables/groups, return "CLEAR".

        Output ONLY "CLEAR" or the clarifying question.
        """
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content'].strip()
            if "CLEAR" in result.upper(): return None
            return result
        except:
            return None

    def interpret_query(self, user_query, column_names, column_values=None):
        """[Agent 2: Planner]"""
        # Build column information string
        col_info = str(column_names)
        if column_values:
            col_info_str = "Available Data Attributes:\n"
            for col, values in column_values.items():
                if col in column_names:
                    # Convert numpy types to Python native types and limit to first 10
                    unique_vals = []
                    for v in list(values)[:10]:
                        # Convert numpy types to native Python types
                        if hasattr(v, 'item'):
                            unique_vals.append(v.item())
                        else:
                            unique_vals.append(str(v))
                    col_info_str += f"  - {col}: {unique_vals}"
                    if len(values) > 10:
                        col_info_str += f" (and {len(values) - 10} more)"
                    col_info_str += "\n"
            col_info = col_info_str
        
        system_prompt = f"""
        You are AI-HOPE. Convert the query into JSON logic.
        {col_info}

        RULES:
        1. Output MUST use one of these 'analysis_type' values: 'case_control', 'survival', 'scan'.
        2. DO NOT use 'descriptive', 'summary', or 'prevalence'. 
           - If user asks for simple counts (e.g. "Check KRAS"), use 'scan' or 'case_control' with 'All' as cohort.
        3. Identify 'target_variable' and 'cohort' definitions.
        4. CRITICAL: The 'target_variable' MUST be an EXACT match from the Available Data Attributes list above.
           - If the user mentions a variable that is NOT in the Available Data Attributes, you MUST still use the exact name they provided (do not substitute with a similar variable).
           - Example: If user says "BRAF_mutation" but only "KRAS_mutation_status" exists, use "BRAF_mutation" as-is (the system will handle the error).
        5. IMPORTANT: Use EXACT values from the Available Data Attributes above. 
           - If user says "late-stage", map to actual Stage values like "Stage III" or "Stage IV" (check what exists in data).
           - If user says "early-stage", map to "Stage I" or "Stage II" (check what exists in data).
           - Always use the exact case and format as shown in the data.
        6. CRITICAL: You MUST ALWAYS return valid JSON. Never return plain text error messages.
           - If you encounter issues, return JSON with an "error" field: {{"error": "description"}}
           - Do NOT return explanatory text outside of JSON format.

        EXAMPLE 1 (Survival Analysis): "Compare survival for KRAS_Status"
        {{"analysis_type": "survival", "group_by": "KRAS_Status"}}
        
        EXAMPLE 2 (Survival Analysis): "Compare survival outcomes between patients with and without KRAS_mutation_status"
        {{"analysis_type": "survival", "group_by": "KRAS_mutation_status"}}

        EXAMPLE 3 (Comparison): "Compare TP53 in Stage IV vs I"
        {{"analysis_type": "case_control", "target_variable": "TP53_Mutation", "case_condition": "TUMOR_STAGE is 'Stage IV'", "control_condition": "TUMOR_STAGE is 'Stage I'"}}

        EXAMPLE 4 (Late vs Early): "KRAS more common in late-stage vs early-stage"
        If data shows TUMOR_STAGE has values ['Stage I', 'Stage II', 'Stage III', 'Stage IV']:
        {{"analysis_type": "case_control", "target_variable": "KRAS_mutation_status", "case_condition": "TUMOR_STAGE is in {{'Stage III', 'Stage IV'}}", "control_condition": "TUMOR_STAGE is in {{'Stage I', 'Stage II'}}"}}

        EXAMPLE 5 (Gender Comparison): "Compare BRAF_mutation frequency in male vs female patients"
        {{"analysis_type": "case_control", "target_variable": "BRAF_mutation", "case_condition": "GENDER is 'male'", "control_condition": "GENDER is 'female'"}}
        Note: Use the exact variable names from the query, even if they don't exist in Available Data Attributes.

        EXAMPLE 6 (Prevalence): "Check KRAS status"
        {{"analysis_type": "scan", "target_variable": "KRAS_mutation_status"}}

        CRITICAL: Return ONLY valid JSON. Use double quotes for all strings. Do not include any text before or after the JSON.
        """

        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ])

        json_text = self._clean_json(response['message']['content'])
        
        # Check if the cleaned text looks like JSON (starts with { or [)
        if not json_text.strip().startswith(('{', '[')):
            # LLM returned plain text instead of JSON - wrap it as an error
            return {"error": f"LLM returned non-JSON response: {json_text[:200]}", "raw_json": json_text[:500]}
        
        return self.verify_logic(json_text, column_names)

    def verify_logic(self, json_str, column_names):
        """[Agent 3: Verifier]"""
        try:
            # Try to parse the JSON
            logic = json.loads(json_str)
            
            # Validate that we got a dictionary
            if not isinstance(logic, dict):
                return {"error": "Parsing failed: Expected JSON object", "raw_json": json_str[:200]}
            
            # Additional safety check: If JSON is malformed or missing critical fields, 
            # it might indicate the query was too vague and should have been caught by Clarifier
            if "error" in logic:
                return logic  # Already has error info

            def fix_col(val, strict=False):
                if val and isinstance(val, str):
                    # First check exact match (case-insensitive)
                    val_lower = val.lower()
                    for col in column_names:
                        if col.lower() == val_lower:
                            return col
                    
                    # If not found and not strict, try fuzzy matching
                    if not strict and val not in column_names:
                        matches = get_close_matches(val, column_names, n=1, cutoff=0.8)  # Increased cutoff for stricter matching
                        if matches:
                            return matches[0]
                return val

            # Fix target_variable with strict matching - it's critical to get right
            if 'target_variable' in logic:
                original_target = logic['target_variable']
                fixed_target = fix_col(original_target, strict=True)
                # If the fixed target doesn't exist in columns, keep original to show error
                if fixed_target not in column_names and original_target not in column_names:
                    # Try one more time with less strict matching but log it
                    fixed_target = fix_col(original_target, strict=False)
                logic['target_variable'] = fixed_target
            
            # Fix group_by variable (used for survival analysis) - also needs correction
            if 'group_by' in logic:
                original_group = logic['group_by']
                fixed_group = fix_col(original_group, strict=False)
                # For survival analysis, group_by is critical - try fuzzy matching
                if fixed_group not in column_names and original_group not in column_names:
                    # Try fuzzy matching with lower cutoff for group_by
                    if isinstance(original_group, str):
                        matches = get_close_matches(original_group, column_names, n=1, cutoff=0.6)
                        if matches:
                            fixed_group = matches[0]
                logic['group_by'] = fixed_group
            return logic
        except json.JSONDecodeError as e:
            # Return more detailed error information
            return {"error": f"Parsing failed: {str(e)}", "raw_json": json_str[:200]}
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}", "raw_json": json_str[:200]}

    def suggest_analysis(self, query):
        prompt = f"""
        Classify this clinical research question into exactly one of these three categories:
        1. 'Survival Analysis': Questions about survival time, outcomes, Kaplan-Meier.
        2. 'Case-Control': Questions comparing groups (e.g., "Stage I vs IV") or Prevalence (e.g. "Count KRAS").
        3. 'Association Scan': Questions asking to find variables correlated/associated with a target, discovery questions, "global scan", "association scan", "find variables correlated with".

        USER QUESTION: "{query}"
        OUTPUT (Return ONLY the category name):
        """
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content'].strip().strip('"').strip("'")
        except:
            return "Error"