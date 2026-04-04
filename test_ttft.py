import time
import httpx

def test_stream_ttft():
    t0 = time.time()
    with httpx.stream("POST", "http://127.0.0.1:8000/query", json={"question": "Hello", "stream": True}, timeout=900) as r:
        r.raise_for_status()
        t_meta = None
        t_ft = None
        
        for line in r.iter_lines():
            if line and t_meta is None:
                t_meta = time.time() - t0
                print(f"Metadata (Retrieval + Reranking) completed in: {t_meta:.2f}s")
            elif line and t_ft is None:
                t_ft = time.time() - t0
                print(f"Time to First LLM Token: {t_ft:.2f}s")
                break

if __name__ == "__main__":
    test_stream_ttft()
