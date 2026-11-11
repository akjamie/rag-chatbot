# Capability: Core Dependency and Model Modernization

## Purpose
Modernize the core dependencies, LangChain framework packages, and partner integrations to LangChain 1.4.9 and LangChain-Core 1.7.6, ensuring high-throughput runnable execution and zero breaking compatibility issues across services.

## ADDED Requirements

### Requirement: Modernize LangChain Ecosystem Dependencies
The application dependencies SHALL specify LangChain 1.4.9, LangChain-Core 1.7.6, LangGraph 1.2.x, and compatible partner packages in `requirements.txt`.

#### Scenario: Verify modern LangChain package resolution
- **GIVEN** a clean Python 3.10+ virtual environment
- **WHEN** dependencies in `requirements.txt` are resolved and installed
- **THEN** `langchain` resolves to 1.4.9, `langchain-core` resolves to 1.7.6, and all model partner integrations install without version conflicts.

**Acceptance criteria**
- [ ] AC1: `requirements.txt` specifies `langchain==1.4.9` and `langchain-core==1.7.6` with compatible partner packages (test: tests/config/test_dependencies.py)

### Requirement: Backward-Compatible Runnable Execution
The query handler, tools, and ReAct agent workflows MUST successfully invoke modern LangChain runnables without encountering deprecated invocation failures.

#### Scenario: Execute chat completion through ReAct agent
- **GIVEN** a query request and an initialized `ReActAgent` using modern LangChain 1.4.x runnables
- **WHEN** the agent receives an input query for document retrieval and response generation
- **THEN** the workflow executes the tool call chain and returns a structured response successfully.

**Acceptance criteria**
- [ ] AC2: Query handler and tool invocation execute without runtime deprecation errors (test: tests/handler/test_generic_query_handler.py)

## Constraints & NFRs
- Service startup time shall not exceed 10 seconds.
- Memory consumption shall remain within existing resource limits under standard loads.

## Out of Scope
- Migrating underlying database architectures or relational schemas.

## Test Strategy
Automated unit and integration testing via pytest. Unit tests verify package resolution and configuration loading. Integration tests in `tests/handler/` verify end-to-end agent query flow, retriever invocations, and history persistence.

