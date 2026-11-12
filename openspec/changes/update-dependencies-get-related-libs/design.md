# Technical Design: Dependency Modernization to LangChain 1.4.9

## Architecture & Component Interaction
The core RAG engine interacts with language models, vector stores, and conversation stores via LangChain 1.4.9 runnables:

```mermaid
flowchart TD
    Client[REST Client / Frontend] -->|HTTP Request| FastAPI[FastAPI App API Gateway]
    FastAPI --> QueryHandler[Generic Query Handler]
    QueryHandler --> ReActAgent[ReAct Agent / LangGraph 1.2.20]
    ReActAgent --> ToolRetriever[Document Retriever / Qdrant 1.1.0]
    ReActAgent --> ModelProvider[LLM Provider: Anthropic / Gemini / Ollama]
    ReActAgent --> ResponseFormatter[Response Formatter]
    QueryHandler --> HistoryStore[Conversation History / Postgres / Redis]
```

## Interface Contracts & Schemas
The REST endpoints (`/api/v1/chat`, `/api/v1/history`) maintain their existing request/response JSON contracts while the internal dependency layer upgrades to LangChain Core 1.7.6 runnables.

## External API Integration & Resilience
- **LangChain Core 1.7.6**: Uses modern standard `Runnable` interfaces (`invoke`, `ainvoke`, `stream`, `astream`).
- **Model Loader Fallbacks**: Offline fallback support for local reranking models (`FlagEmbedding`, cross-encoder).
- **Deferred Resilience**: Custom exponential-jitter backoff retry wrappers for remote LLM calls are explicitly deferred to a future resilience change.

## Security & Operational NFRs
- All secrets and API keys loaded securely via environment variables (`python-dotenv`).
- Pydantic v2 data validation for request payloads.
- Cold startup time for FastAPI uvicorn workers `<= 10s` (aligned with `spec.md`).
