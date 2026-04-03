.PHONY: install api ui test lint clean docker

# ── Local dev ─────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run streamlit_app/Home.py --server.port 8501

# Run both concurrently (requires a POSIX shell or use two terminals on Windows)
dev:
	@echo "Start the API: make api"
	@echo "Start the UI:  make ui"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	ruff check app/ streamlit_app/ tests/

format:
	ruff format app/ streamlit_app/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf app/storage/vectordb app/storage/logs app/storage/files

# ── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# ── Seed ─────────────────────────────────────────────────────────────────────
seed:
	python scripts/ingest_sample.py

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	python scripts/run_evaluation.py
