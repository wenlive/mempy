# mempy API Reference

Complete API reference for the mempy memory management library.

## Table of Contents

- [Memory](#memory)
- [Embedder](#embedder)
- [MemoryProcessor](#memoryprocessor)
- [RelationType](#relationtype)
- [Data Classes](#data-classes)
- [Exceptions](#exceptions)

---

## Memory

The main API for memory management.

### Constructor

```python
mempy.Memory(
    embedder: Embedder,
    processor: Optional[MemoryProcessor] = None,
    storage_path: Optional[str] = None,
    verbose: bool = False
) -> Memory
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `embedder` | `Embedder` | Yes | - | User-provided embedder instance |
| `processor` | `MemoryProcessor` | No | `None` | Optional intelligent processor |
| `storage_path` | `str` | No | `~/.mempy/data` | Custom storage path |
| `verbose` | `bool` | No | `False` | Enable verbose logging |

### Methods

#### add()

Add a new memory with optional intelligent processing.

```python
async add(
    content: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]
```

**Returns:** The memory ID if successful, `None` if skipped by processor.

**Example:**
```python
memory_id = await memory.add(
    "I prefer working remotely",
    user_id="alice",
    metadata={"source": "chat"}
)
```

#### search()

Search for memories by semantic similarity.

```python
async search(
    query: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: int = 10
) -> List[Memory]
```

**Returns:** List of memories ranked by similarity.

**Example:**
```python
results = await memory.search("remote work", user_id="alice")
for m in results:
    print(f"{m.memory_id}: {m.content}")
```

#### get()

Get a specific memory by ID.

```python
async get(memory_id: str) -> Optional[Memory]
```

**Returns:** The memory if found, `None` otherwise.

#### get_all()

Get all memories matching filters.

```python
async get_all(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Memory]
```

**Returns:** List of matching memories.

#### update()

Update an existing memory's content.

```python
async update(memory_id: str, content: str) -> str
```

**Returns:** The memory ID.

**Raises:** `StorageError` if memory not found.

#### delete()

Delete a memory by ID.

```python
async delete(memory_id: str) -> None
```

**Raises:** `StorageError` if deletion fails.

#### delete_all()

Delete all memories for a user.

```python
async delete_all(user_id: str) -> None
```

#### add_relation()

Add a relation between two memories.

```python
async add_relation(
    from_id: str,
    to_id: str,
    relation_type: RelationType,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

**Example:**
```python
await memory.add_relation(
    mem1,
    mem2,
    mempy.RelationType.PROPERTY_OF
)
```

#### get_relations()

Get relations for a memory.

```python
async get_relations(
    memory_id: str,
    direction: str = "both",
    max_depth: int = 2
) -> List[Relation]
```

**Parameters:**
- `direction`: `"out"`, `"in"`, or `"both"`
- `max_depth`: Maximum depth to traverse in graph

#### reset()

Clear all data.

```python
async reset() -> None
```

---

## Embedder

Abstract interface for embedding generation.

### Required Methods

```python
class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        pass
```

### Implementation Example

```python
import aiohttp
from typing import List
from mempy import Embedder

class MyEmbedder(Embedder):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/embed",
                json={"text": text}
            ) as resp:
                data = await resp.json()
                return data["embedding"]
```

---

## MemoryProcessor

Abstract interface for intelligent memory processing.

### Required Methods

```python
class MemoryProcessor(ABC):
    @abstractmethod
    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        """
        Decide what operation to perform.

        Returns:
            ProcessorResult with action ("add"/"update"/"delete"/"none"),
            memory_id (for update/delete), content (for update), and reason.
        """
        pass
```

### ProcessorResult

```python
@dataclass
class ProcessorResult:
    action: str  # "add" | "update" | "delete" | "none"
    memory_id: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None
```

---

## RelationType

Relation types based on relational algebra concepts.

| Type | Description |
|------|-------------|
| `RELATED` | General correlation |
| `EQUIVALENT` | Equivalence |
| `CONTRADICTORY` | Contradiction |
| `GENERALIZATION` | Superclass relation |
| `SPECIALIZATION` | Subclass relation |
| `PART_OF` | Part-whole composition |
| `PRECEDES` | Temporal: comes before |
| `FOLLOWS` | Temporal: comes after |
| `CAUSES` | Causality |
| `CAUSED_BY` | Reverse causality |
| `SIMILAR` | Similarity |
| `PROPERTY_OF` | Attribute/property |
| `INSTANCE_OF` | Instance relation |
| `CONTEXT_FOR` | Context/background |

---

## Data Classes

### Memory

```python
@dataclass
class Memory:
    memory_id: str
    content: str
    embedding: List[float]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    priority: float = 0.5
    confidence: float = 1.0
```

### Relation

```python
@dataclass
class Relation:
    relation_id: str
    from_id: str
    to_id: str
    type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
```

---

## Exceptions

| Exception | Description |
|-----------|-------------|
| `MempyError` | Base exception for all mempy errors |
| `EmbedderError` | Embedder operation failed |
| `StorageError` | Storage operation failed |
| `ProcessorError` | Processor operation failed |
