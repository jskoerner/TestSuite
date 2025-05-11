import requests
import time
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"  # Update if needed
APP_NAME = "enhanced_model_agent"
USER_ID = "test1"
QUESTIONS = [
    "What are the benefits of vitamin C?",
    "What foods are high in iron?",
    "How much water should I drink daily?",
    # Add more questions as needed
]
OUTPUT_FILE = "rag_batch_test_results.json"


def send_question(question):
    url = f"{BASE_URL}/api/conversation/{APP_NAME}/{USER_ID}"
    payload = {"message": question}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def main():
    results = []
    for question in QUESTIONS:
        agent_start = time.time()
        agent_start_iso = datetime.utcnow().isoformat() + "Z"
        try:
            response = send_question(question)
            agent_end = time.time()
            agent_end_iso = datetime.utcnow().isoformat() + "Z"
            elapsed = agent_end - agent_start

            # Extract answer and confidence
            answer = ""
            retriever_confidence = None
            last_user_message = None
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("response") or ""
                state = response.get("state", {})
                retriever_confidence = state.get("retriever_confidence")
                last_user_message = state.get("last_user_message")

            timing_info = {
                "agent_start": agent_start_iso,
                "agent_end": agent_end_iso,
                "elapsed": elapsed
            }
            state_info = {
                "retriever_confidence": retriever_confidence,
                "last_user_message": last_user_message
            }
            results.append({
                "question": question,
                "answer": answer,
                "state": state_info,
                "timing": timing_info
            })
            print(f"Q: {question}\nA: {answer}\nConfidence: {retriever_confidence}\nElapsed: {elapsed:.2f}s\n---")
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            results.append({
                "question": question,
                "error": str(e)
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
