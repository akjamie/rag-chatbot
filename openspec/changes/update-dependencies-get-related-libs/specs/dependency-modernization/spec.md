# Capability: Core Dependency and Model Modernization

## Purpose
Modernize the core dependencies, LangChain framework packages, and partner integrations to LangChain 1.4.9 and LangChain-Core 1.7.6, ensuring deterministic dependency pinning, deprecation-free runnable execution, and verified startup and reranker compatibility.

## ADDED Requirements

### Requirement: Modernize LangChain Ecosystem Dependencies
The application dependencies SHALL specify exact-pinned versions in `requirements.txt` for the LangChain ecosystem (`langchain==1.4.9`, `langchain-core==1.7.6`, `langchain-community==0.4.2`, `langgraph==1.2.20`, `langchain-huggingface==1.2.3`, `langchain-qdrant==1.1.0`, `langchain-anthropic==1.7.6`, `langchain-google-genai==4.4.3`, `langchain-ollama==1.1.0`) and minimum-bounded runtime drivers (`fastapi>=0.115.6`, `uvicorn>=0.32.1`, `neo4j>=5.26.0`, `spacy>=3.8.4`).

#### Scenario: Verify modern LangChain package resolution
- **GIVEN** a clean Python 3.10+ virtual environment
- **WHEN** `requirements.txt` is parsed and validated against the target dependency manifest
- **THEN** all nine LangChain ecosystem packages match their exact pinned versions and runtime drivers meet minimum version bounds without conflicts.

**Acceptance criteria**
- [ ] AC1: `requirements.txt` pins `langchain==1.4.9`, `langchain-core==1.7.6`, `langchain-community==0.4.2`, `langgraph==1.2.20`, `langchain-huggingface==1.2.3`, `langchain-qdrant==1.1.0`, `langchain-anthropic==1.7.6`, `langchain-google-genai==4.4.3`, and `langchain-ollama==1.1.0`, plus `fastapi>=0.115.6`, `uvicorn>=0.32.1`, `neo4j>=5.26.0`, `spacy>=3.8.4` (test: tests/config/test_dependencies.py)

### Requirement: Backward-Compatible Runnable Execution and Reranker Validation
The query handler, tools, local reranker (`FlagEmbedding` / cross-encoder), and ReAct agent workflows MUST execute modern LangChain 1.4.9 runnables without `DeprecationWarning` or runtime failures, and the FastAPI app MUST complete cold startup within 10 seconds.

#### Scenario: Execute chat completion through ReAct agent without deprecation warnings
- **GIVEN** a query request and an initialized `ReActAgent` using modern LangChain 1.4.9 runnables
- **WHEN** the handler test suite executes with `pytest -W error::DeprecationWarning`
- **THEN** the workflow executes the tool call chain and returns a structured response with zero `DeprecationWarning` exceptions.

#### Scenario: Verify FastAPI cold startup and local reranker loading
- **GIVEN** the FastAPI application and local reranker configuration (`FlagEmbedding` / cross-encoder)
- **WHEN** the app initializes and loads the reranker model interface
- **THEN** cold startup completes in `<= 10` seconds and reranker scoring returns valid float scores.

**Acceptance criteria**
- [ ] AC2: Handler and tool tests pass under `pytest -W error::DeprecationWarning` without LangChain deprecation errors (test: tests/handler/test_generic_query_handler.py)
- [ ] AC3: FastAPI application cold startup completes in `<= 10` seconds (test: tests/config/test_startup.py)
- [ ] AC4: Local reranker (`FlagEmbedding` / cross-encoder) initializes and scores document candidates without import or runtime errors (test: tests/handler/test_reranker.py)

## Constraints & NFRs
- Service cold startup time SHALL NOT exceed 10 seconds (aligned with `design.md`).
- Memory footprint monitoring under production load is deferred to container-level observability metrics (out of unit test scope).

## Out of Scope
- Migrating underlying database architectures or relational schemas.
- Adding custom exponential-jitter retry wrappers for remote LLM calls.

## Test Strategy
Automated unit and integration testing via pytest. `tests/config/test_dependencies.py` verifies exact dependency pins. `tests/handler/test_generic_query_handler.py` runs with `-W error::DeprecationWarning` to deterministically catch LangChain 1.x deprecation regressions. `tests/config/test_startup.py` measures FastAPI app import/startup latency (`<= 10s`), and `tests/handler/test_reranker.py` validates local reranker compatibility.
