# mempy

A memory management library with vector and graph storage, similar to [mem0](https://github.com/mem0ai/mem0).

## 📊 Research Report

**[Agent Memory 深度研究报告](Agent_Memory_深度研究报告.md)** - 基于对LangGraph、M3-Agent、PowerMem等前沿框架的深度调研，从数据库系统设计原则出发，对Agent Memory进行系统性重构分析。

**核心观点**：
- **演化（Evolution）是核心瓶颈**：记忆不是静态存储，而是动态的ETL过程
- **"行为主义"优于"符号主义"**：提出基于共现关系的Memory Transition Graph（使用轨迹图）
- **多模态的锚点效应**：高模态数据（语音/图像）天然充当Hub，是非结构化文本建立关联的关键
- **数据库范式的冲突**：传统强Schema约束与LLM模糊性的根本矛盾

---

## Features

- **Zero Configuration**: pip install + import = ready to use
- **Vector + Graph**: Dual first-class citizens for semantic search and relation reasoning
- **Pluggable**: User-provided embedder, optional LLM processor
- **Async API**: Full async/await support
- **Observable**: Built-in logging and operation tracking
- **Strategy System**: Optional intelligent processing and automatic relation building

## Strategy System

mempy provides a flexible strategy system for advanced users to customize memory behavior:

### MemoryProcessor (Optional)

Intelligently decide what to do with new content (add, update, delete, or skip):

```python
from mempy.core.interfaces import MemoryProcessor, ProcessorResult

class MyProcessor(MemoryProcessor):
    async def process(self, content, existing_memories):
        # Your logic here
        if len(content.strip()) < 5:
            return ProcessorResult(action="none", reason="Too short")
        return ProcessorResult(action="add", content=content, reason="New content")

memory = mempy.Memory(embedder=MyEmbedder(), processor=MyProcessor())
```

### RelationBuilder (Optional)

Automatically construct graph relations when adding memories:

```python
from mempy.strategies import RandomRelationBuilder

builder = RandomRelationBuilder(max_relations=3)
memory = mempy.Memory(embedder=MyEmbedder(), relation_builder=builder)
# Relations are automatically built during memory.add()
```

**Learn More**: See [doc/strategy_system.md](doc/strategy_system.md) for detailed documentation.

## Installation

```bash
pip install chromadb networkx aiohttp
```

## Quick Start

```python
import asyncio
import mempy
from typing import List

# 1. Implement the Embedder interface (required)
class MyEmbedder(mempy.Embedder):
    def __init__(self):
        self._dimension = 768  # Must declare dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # Call your LLM service (local or remote)
        return await my_llm_service.embed(text)

# 2. Create Memory instance
async def main():
    memory = mempy.Memory(
        embedder=MyEmbedder(),
        verbose=True  # Enable logging
    )

    # 3. Add memories
    mem1 = await memory.add("I like blue", user_id="alice")
    mem2 = await memory.add("Alice works at Google", user_id="alice")

    # 4. Search
    results = await memory.search("Alice's job", user_id="alice")
    for r in results:
        print(f"{r.memory_id}: {r.content}")

    # 5. Add relations
    await memory.add_relation(
        mem1,
        mem2,
        mempy.RelationType.PROPERTY_OF
    )

asyncio.run(main())
```

## Configuration

Storage path defaults to `~/.mempy/data`. Override with:

```bash
export MEMPY_HOME=/custom/path
```

Or:

```python
memory = mempy.Memory(embedder=embedder, storage_path="/my/path")
```

## API Reference

### Main Methods

| Method | Description |
|--------|-------------|
| `add(content, user_id, ...)` | Add a new memory |
| `search(query, user_id, limit)` | Semantic search |
| `get(memory_id)` | Get specific memory |
| `get_all(user_id)` | Get all memories for user |
| `update(memory_id, content)` | Update memory content |
| `delete(memory_id)` | Delete a memory |
| `delete_all(user_id)` | Delete all user memories |
| `add_relation(from, to, type)` | Add relation between memories |
| `get_relations(memory_id)` | Get relations for a memory |
| `reset()` | Clear all data |

### Relation Types

Based on relational algebra concepts:

```python
mempy.RelationType.RELATED         # Correlation
mempy.RelationType.EQUIVALENT      # Equivalence
mempy.RelationType.CONTRADICTORY   # Contradiction
mempy.RelationType.GENERALIZATION  # Generalization
mempy.RelationType.SPECIALIZATION  # Specialization
mempy.RelationType.PART_OF         # Part-whole
mempy.RelationType.PRECEDES        # Temporal: comes before
mempy.RelationType.FOLLOWS         # Temporal: comes after
mempy.RelationType.CAUSES          # Causality
mempy.RelationType.CAUSED_BY       # Reverse causality
mempy.RelationType.SIMILAR         # Similarity
mempy.RelationType.PROPERTY_OF     # Attribute/property
mempy.RelationType.INSTANCE_OF     # Instance relation
mempy.RelationType.CONTEXT_FOR     # Context relation
```

## Project Structure

```
mempy/
├── core/          # Data classes and interfaces
├── storage/       # Vector + graph storage backends
├── processors/    # Optional LLM processors
├── memory.py      # Main user API
└── config.py      # Configuration
```

## Documentation

### 📖 User Guide

- **[Quick Start](doc/quickstart.md)** - Get started in 5 minutes

### 🔧 Developer Documentation

- **[API Reference](doc/api.md)** - Complete API documentation
- **[Architecture](doc/architecture.md)** - Architecture and design (中文)
- **[Adapter Guide](doc/adapter-guide.md)** - Creating custom embedders
- **[Strategy System](doc/strategy_system.md)** - Pluggable strategy architecture (NEW)
- **[RelationBuilder Guide](doc/relation_builder_guide.md)** - Automatic relation building (NEW)
- **[Strategies Guide](doc/strategies.md)** - Custom evolution strategies
- **[Benchmark Guide](doc/benchmark.md)** - Running evaluations (UPDATED)

More documentation: [doc/README.md](doc/README.md)

## License

MIT
