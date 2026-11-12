# Proposal: Modernize Core Dependencies and Models (LangChain 1.4.9)

## Why
The backend services are currently pinned to earlier, legacy versions of LangChain (`0.3.7`), LangGraph (`0.2.48`), and associated model partner packages. Upgrading to the actively maintained LangChain `1.4.9` and `langchain-core` `1.7.6` release line ensures upstream security-patch maintenance, enhances LCEL streaming performance, improves model provider integrations (Google GenAI, Anthropic, HuggingFace, Ollama, Qdrant), and brings modern runnable stability to the multi-agent RAG workflow.

## What Changes
- Upgrade core orchestration and model integration packages in `requirements.txt` (exact-pinned `==` for reproducible LangChain ecosystem compatibility):
  - `langchain==1.4.9`
  - `langchain-core==1.7.6`
  - `langchain-community==0.4.2`
  - `langgraph==1.2.20`
  - `langchain-huggingface==1.2.3`
  - `langchain-qdrant==1.1.0`
  - `langchain-anthropic==1.7.6`
  - `langchain-google-genai==4.4.3`
  - `langchain-ollama==1.1.0`
- Align backend runtime dependencies (minimum-bounded `>=` for backward-compatible ASGI/DB drivers):
  - `fastapi>=0.115.6`, `uvicorn>=0.32.1`, `neo4j>=5.26.0`, `spacy>=3.8.4`
- Audit and adapt code call sites for any deprecated interfaces between LangChain 0.3 and 1.4.x, verifying runnable execution, memory stores, and vector search.
- Validate local model integrations (reranker `FlagEmbedding`, cross-encoder) against updated HuggingFace packages.
- Add `tests/config/test_dependencies.py` and `tests/config/test_startup.py` in the implementation PR.

## Capabilities
### Modernized RAG & Agent Pipeline
- High-performance execution with LangChain 1.4.9 runnable architecture.
- Full compatibility with newer model providers and Qdrant vector retrieval.

## Impact
- **Codebase**: `requirements.txt`, model loaders, agent invocation handlers, test suites.
- **Runtime**: FastAPI service bootstrapping and background tasks.
- **Tests**: Pytest suites in `tests/` covering dependency pinning, startup, reranker loading, generic query handler, tools, and conversation history.

## Out of Scope
- Rewriting core business logic or switching vector databases away from Qdrant/Postgres/Neo4j.
- Changing REST API contracts exposed to the frontend.
- Adding new exponential-jitter retry middleware for remote LLM calls (deferred to a dedicated resilience change).

## Dependencies & Assumptions
- Python >= 3.10 runtime environment.
- Internet access or pre-cached wheels for dependency installation.
