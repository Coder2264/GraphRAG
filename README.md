# Thinking on Graph

A highly modular, swapping-friendly FastAPI backend designed to experiment with different LLMs, embedders, and Retrieval-Augmented Generation (RAG) approaches.

Currently, it supports three distinct query modes:
1. **GraphRAG**: Traverses a knowledge graph (e.g. Neo4j) to build context.
2. **Standard RAG**: Uses vector similarity search (e.g. pgvector) over document chunks.
3. **None-RAG**: A direct pass-through to the LLM (useful as a baseline).

---

## 🏗 Directory Structure

```text
app/
├── config.py                 # Pydantic Settings (reads .env — purely for keys/DSNs)
├── registry.py               # Provider registry mapping strings (e.g. "neo4j") to Classes
├── dependencies.py           # FastAPI dependency injection helpers
├── factory.py                # ServiceFactory — parses config & registry to build instances
│
├── core/                     # 🛡️ Contracts: Abstract Base Classes (ABCs)
│   ├── llm.py                # BaseLLM (exposes `generate`, `generate_structured`)
│   ├── embedder.py           # BaseEmbedder
│   ├── graph_store.py        # BaseGraphStore
│   ├── vector_store.py       # BaseVectorStore
│   ├── retriever.py          # BaseRetriever (GraphRAG vs RAG)
│   └── ingestion.py          # BaseIngestionPipeline
│
├── implementations/          # ⚙️ Concrete Classes (implement the core/ ABCs)
│   ├── in_memory/            # Zero-dependency, ephemeral stubs (for dev/test)
│   ├── neo4j/                # Production graph store implementation
│   └── postgres/             # Production vector store implementation (pgvector)
│
├── models/                   # 📦 Pydantic request/response schemas
│   ├── ingest.py             
│   └── query.py              
│
├── services/                 # 🧠 Business Logic Use Cases
│   ├── ingestion_service.py  # Orchestrates ingestion without knowing about dependencies
│   └── query_service.py      # Executes queries via injected Retrievers + LLMs
│
└── api/
    └── v1/                   # 🌐 HTTP Routing Layer
        ├── ingest.py         
        ├── query.py          
        └── router.py         # Main APIRouter combining ingest and query endpoints
```

---

## 🧩 Architectural Principles (SOLID)

This codebase is specifically structured so that you can **test lots of different LLMs and new approaches** without tearing apart the routing or orchestration logic.

* **Single Responsibility (SRP)**: Routes only handle HTTP in/out. Services only own the use-case logic. Implementations only know how to talk to their specific backend.
* **Open/Closed (OCP)**: You can add an `AnthropicLLM` or a `QdrantVectorStore` by simply adding a new file and updating `app/registry.py`. You will never need to modify existing files to add new providers. 
* **Liskov Substitution (LSP)**: Every implementation strictly inherits from its respective abstract class in `app/core/`. A `QueryService` does not know or care if it's using an `InMemoryLLM` or an `OpenAILLM`.
* **Interface Segregation (ISP)**: Interfaces are kept intentionally narrow. 
* **Dependency Inversion (DIP)**: `app/dependencies.py` and `app/factory.py` handle wiring concrete classes together and injecting them into the services. The application routes explicitly depend on interfaces, never on concretions.

---

## 🛠 Adding New Features & Providers

The architecture heavily relies on the Open/Closed Principle (OCP). You should be able to add new functionality without touching existing business logic.

### 1. Adding a New Model Provider (LLM, Embedder, Graph/Vector Store)

If you want to add a real LLM integration (e.g., OpenAI) or a database:

1. **Files to Add**: Create a new file for the implementation.
   * *Example*: `app/implementations/openai/llm.py`
2. **Implement the Contract**: Inherit from the relevant `app/core/` base class.
   ```python
   from app.core.llm import BaseLLM
   
   class OpenAILLM(BaseLLM):
       async def generate(self, prompt: str, context: str = "") -> str:
           # call OpenAI API
           ...
   ```
3. **Files to Modify (Slightly)**:
   * **`app/registry.py`**: Import your new class and add it to the corresponding registry dictionary (e.g., `LLM_REGISTRY`, `VECTOR_STORE_REGISTRY`). 
   * **`app/config.py`**: Add any necessary environment variables (e.g., `openai_api_key`) to the `Settings` class so they can be loaded from `.env`.
   * **`app/factory.py`** (Optional): If your new provider requires specific initialization arguments (like API keys or specific connection strings that go beyond a simple `__init__()`), update the respective `_build_*` method in `ServiceFactory`.
   * **`.env`** (Optional): Update your `.env` to point your `default_llm` to `"openai"`.
4. **Files to Leave Untouched**:
   * **`app/core/*`**: The abstract base classes don't care about your new provider.
   * **`app/services/*`**: The business logic dynamically receives whichever provider is registered.
   * **`app/api/*`**: The routing layer doesn't need to know.

### 2. Adding a New Query Mode or Retrieval Strategy

If you want to add a new RAG strategy (e.g., Hybrid Search):

1. **Files to Add**: Create a new Retriever class that inherits from `BaseRetriever` in `app/core/retriever.py`. You can place it in `app/implementations/in_memory/retrievers.py` or a new dedicated module if it grows large.
2. **Files to Modify**:
   * **`app/models/query.py`**: Add the new mode to the `QueryMode` Enum.
   * **`app/factory.py`**: Update `ServiceFactory.get_retriever(mode)` to return an instance of your new retriever when the new `QueryMode` is requested.
3. **Files to Leave Untouched**:
   * **`app/services/query_service.py`**: It just calls `retriever.retrieve()` generically.
   * **`app/api/v1/query.py`**: The route just passes the chosen enum mode down.

### 3. Adding Entirely New API Features (e.g., Document Management)

1. **Files to Add**:
   * **Models**: Add request/response Pydantic schemas in `app/models/<feature>.py`.
   * **Services**: Create the business logic in `app/services/<feature>_service.py`.
   * **Routes**: Create the endpoints in `app/api/v1/<feature>.py`.
2. **Files to Modify**:
   * **`app/api/v1/router.py`**: Include the new router (`router.include_router(...)`).
3. **Files to Leave Untouched**: Almost everything else!

---

## 🚀 Running the Server

Make sure your virtual environment is active and dependencies are installed.

```bash
# 1. Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
uvicorn main:app --reload
```

Then visit the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
