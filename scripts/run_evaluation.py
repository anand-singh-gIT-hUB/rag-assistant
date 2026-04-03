"""
scripts/run_evaluation.py
──────────────────────────
CLI script to trigger Ragas evaluation via the API and print results.
Run: python scripts/run_evaluation.py
"""
import httpx
import json


def main():
    with httpx.Client(timeout=300) as client:
        print("Triggering evaluation…")
        r = client.post("http://localhost:8000/evaluate/run")
        r.raise_for_status()
        result = r.json()

    print(f"\nRun ID: {result['run_id']}")
    print(f"Questions: {result['num_questions']}")
    print(f"Status: {result['status']}")
    print("\nMetrics:")
    for metric, value in result["metrics"].items():
        print(f"  {metric:30s}: {value:.4f}")


if __name__ == "__main__":
    main()
