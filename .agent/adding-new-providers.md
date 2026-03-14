# Adding New Providers

The registry pattern means adding a provider never requires touching services or route handlers. Follow the steps below for each provider type.

---

## 1. New LLM Provider (e.g. OpenAI)

### Step 1 — Create the implementation file

```
app/implementations/openai/llm.py
```

```python
"""
OpenAI LLM implementation.

LSP: Fully substitutable for BaseLLM.
"""

from __future__ import annotations

import openai

from app.core.llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(
        self, prompt: str, context: str = "", system_prompt: str = ""
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model, messages=messages
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self, prompt: str, response_model: type, context: str = "", system_prompt: str = ""
    ) -> object:
        # Use instructor or manual JSON parsing here
        raise NotImplementedError
```

### Step 2 — Register the key

In `app/registry.py`:

```python
from app.implementations.openai.llm import OpenAILLM

LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "in_memory": InMemoryLLM,
    "ollama":    OllamaLLM,
    "openai":    OpenAILLM,   # ← add this line
}
```

### Step 3 — Wire constructor args in factory

In `app/factory.py`, inside `_build_llm`:

```python
def _build_llm(self) -> BaseLLM:
    cls = LLM_REGISTRY[self._llm_key]
    if self._llm_key == "ollama":
        return cls(model_name=settings.ollama_llm_model, base_url=settings.ollama_base_url)
    if self._llm_key == "openai":
        return cls(api_key=settings.openai_api_key, model=settings.openai_llm_model)
    return cls()
```

### Step 4 — Add config fields

In `app/config.py`:
```python
openai_llm_model: str = "gpt-4o"
```

In `.env.example`:
```
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o
```

### Step 5 — Activate

In `.env`:
```
DEFAULT_LLM=openai
```

---

## 2. New Embedder (e.g. OpenAI Embeddings)

Same pattern — subclass `BaseEmbedder`, register in `EMBEDDER_REGISTRY`, wire in `_build_embedder`.

---

## 3. New Vector Store (e.g. Qdrant)

Subclass `BaseVectorStore`. Implement the full lifecycle (`connect`, `close`) and all abstract methods (`upsert`, `delete`, `search`, `get`). Register in `VECTOR_STORE_REGISTRY`, wire in `_build_vector_store`.

---

## 4. New Graph Store (e.g. ArangoDB)

Subclass `BaseGraphStore`. Implement all abstract methods. Register in `GRAPH_STORE_REGISTRY`, wire in `_build_graph_store`.

---

## Rules

- The implementation file and its `__init__.py` are the **only** files you should need to create.
- `registry.py` and `factory.py` are the **only** existing files you should need to modify.
- Do not modify `QueryService`, `IngestionService`, or any `app/core/` file.
- Every new config field must be added to **both** `app/config.py` and `.env.example`.
