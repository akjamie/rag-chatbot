# Tasks: update-dependencies-get-related-libs

## 1. Dependency Modernization
- [x] 1.1 [TEST] Add test asserting exact pins for LangChain ecosystem and minimum bounds for runtime drivers in requirements.txt (`tests/config/test_dependencies.py`)
  - verify: `pytest --noconftest tests/config/test_dependencies.py` -> PASS (2 passed in 0.02s)
- [x] 1.2 [IMPL] Update `requirements.txt` to pin LangChain 1.4.9 ecosystem and bounded drivers
  - evidence: `pip install -r requirements.txt` -> successfully resolved and installed exact pins

## 2. Startup & Reranker Verification
- [x] 2.1 [TEST] Add FastAPI application cold startup latency test asserting import/init <= 10s (`tests/config/test_startup.py`)
  - verify: `pytest --noconftest tests/config/test_startup.py` -> PASS (1 passed in 7.86s <= 10.0s)
- [x] 2.2 [TEST] Add local reranker initialization and scoring validation test (`tests/handler/test_reranker.py`)
  - verify: `pytest --noconftest tests/handler/test_reranker.py` -> PASS (2 passed in 0.05s)
- [x] 2.3 [IMPL] Verify and adapt handler, tools, and reranker runnables under LangChain 1.4.9 without deprecation warnings (`tests/handler/test_generic_query_handler.py`)
  - evidence: `pytest -v` -> PASS (23 passed in 11.00s)
