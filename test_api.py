import requests
import json

try:
    print("Testing Glossary Term...")
    resp = requests.post("http://127.0.0.1:8000/api/chat", json={"message": "what is ebitda"})
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")

    print("\nTesting LLM Fallback (might fail if Ollama not running)...")
    resp2 = requests.post("http://127.0.0.1:8000/api/chat", json={"message": "hello"})
    print(f"Status: {resp2.status_code}")
    print(f"Response: {json.dumps(resp2.json(), indent=2)}")

except Exception as e:
    print(f"Test failed: {e}")
