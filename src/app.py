from llm_agent import AIHOPEAgent
from data_loader import load_dataset
from analysis_engine import run_analysis

if __name__ == "__main__":
    agent = AIHOPEAgent()
    print("🧠  AI-HOPE ready. Ask a research question.")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        intent = agent.interpret_query(user_input)
        print("Parsed intent:", intent.model_dump())
        df = load_dataset(f"data/{intent.dataset}")
        results = run_analysis(df, intent)
        print(results)
