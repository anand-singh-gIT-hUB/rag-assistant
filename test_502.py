import httpx
import sys

def test_ollama():
    print("Testing Ollama with phi3...")
    try:
        r = httpx.post("http://localhost:11434/api/generate", json={
            "model": "phi3",
            "prompt": "Hello",
            "stream": False
        }, timeout=120.0)
        print(f"Ollama Status: {r.status_code}")
        if r.status_code != 200:
            print("Ollama Error Body:", r.text)
        else:
            print("Ollama responded successfully!")
    except Exception as e:
        print(f"Ollama connection failed: {e}")

def test_backend_query():
    print("\nTesting FastAPI Backend /query...")
    try:
        r = httpx.post("http://localhost:8000/query", json={
            "question": "Hello",
            "top_k": 5,
            "rerank": True,
            "doc_ids": []
        }, timeout=180.0)
        print(f"Backend Status: {r.status_code}")
        if r.status_code != 200:
            print("Backend Error Body:", r.text)
    except Exception as e:
        print(f"Backend connection failed: {e}")

if __name__ == "__main__":
    test_ollama()
    test_backend_query()
