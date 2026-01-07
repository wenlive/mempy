# 策略系统架构文档

## 概述

mempy策略系统提供了可插拔的记忆处理和关系构建机制，允许高级用户自定义记忆系统的行为。

策略系统是**完全可选的**。如果不使用任何策略，mempy会以简单直接的方式工作：
- 所有内容都会被保存
- 不会自动创建任何关系
- 保持向后兼容性

## 设计理念

### 三阶段处理流水线

Memory类的`add()`方法采用三阶段处理流水线：

```
1. Ingest Strategy (Processor) → 决定如何处理新内容
2. Storage → 持久化存储记忆
3. Graph Strategy (RelationBuilder) → 自动构建图关系
```

### 设计原则

1. **可选性**：所有策略都是可选的，默认行为简单直接
2. **显式调用**：策略调用在代码中清晰可见，使用明确的注释标记
3. **封装性**：策略逻辑完全封装在私有方法中（`_apply_processor_strategy()`和`_apply_relation_builder_strategy()`）
4. **可扩展性**：用户可实现自定义策略接口来扩展功能
5. **错误容忍**：策略失败时有合理的降级处理，不会影响核心存储功能

## 核心组件

### 1. Processor Strategy（摄取策略）

#### MemoryProcessor接口

`MemoryProcessor`是一个抽象接口，用于决定如何处理新添加的内容。

```python
from mempy.core.interfaces import MemoryProcessor
from mempy.core.memory import ProcessorResult

class MyProcessor(MemoryProcessor):
    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        """
        处理新内容，决定操作类型。

        Args:
            content: 待添加的内容
            existing_memories: 已存在的相关记忆列表

        Returns:
            ProcessorResult: 包含action, memory_id, content, reason
        """
        # 你的逻辑
        return ProcessorResult(
            action="add",  # "add", "update", "delete", "none"
            memory_id=None,
            content=content,
            reason="原因说明"
        )
```

#### 支持的操作类型

- **`add`**: 添加为新记忆（默认）
- **`update`**: 更新已存在的记忆（需提供memory_id）
- **`delete`**: 删除已存在的记忆（需提供memory_id）
- **`none`**: 跳过，不保存内容

#### 错误处理

如果Processor抛出异常，系统会自动降级为`add`操作，确保内容不会被丢失。

---

### 2. RelationBuilder Strategy（图构建策略）

#### RelationBuilder接口

`RelationBuilder`是一个抽象接口，用于在添加新记忆时自动构建图关系。

```python
from mempy.strategies import RelationBuilder
from mempy.core.memory import Memory, RelationType
from typing import List, Tuple, Dict, Any

class MyRelationBuilder(RelationBuilder):
    async def build(
        self,
        new_memory: Memory,
        existing_memories: List["Memory"]
    ) -> List[Tuple[str, str, RelationType, Dict[str, Any]]]:
        """
        为新记忆构建关系到已存在的记忆。

        Args:
            new_memory: 新添加的记忆对象
            existing_memories: 已存在的记忆列表

        Returns:
            List of tuples: (from_id, to_id, relation_type, metadata)
        """
        relations = []
        for existing in existing_memories:
            # 你的逻辑
            if should_connect(new_memory, existing):
                relations.append((
                    new_memory.memory_id,
                    existing.memory_id,
                    RelationType.RELATED,
                    {"confidence": 0.9}
                ))
        return relations
```

#### 内置实现

**RandomRelationBuilder**：随机关系构建器，用于测试和演示。

```python
from mempy.strategies import RandomRelationBuilder, RelationType

builder = RandomRelationBuilder(
    max_relations=3,  # 每个新记忆最多创建3个关系
    relation_types=[RelationType.RELATED, RelationType.SIMILAR],
    build_probability=1.0,  # 100%概率构建关系
    seed=42  # 随机种子，用于可重现性
)
```

#### 错误处理

如果RelationBuilder抛出异常，系统会记录警告但不影响记忆的存储。

---

## 实现细节

### Memory类中的策略调用

Memory类的`add()`方法清晰地展示了三阶段流水线：

```python
async def add(self, content: str, ...):
    # ========== STRATEGY 1: Ingest (Processor) ==========
    decision = await self._apply_processor_strategy(content, user_id)

    # 处理非add操作
    if decision.action == "update" and decision.memory_id:
        return await self.update(...)
    elif decision.action == "delete" and decision.memory_id:
        await self.delete(...)
        return None
    elif decision.action == "none":
        return None

    # ========== STORAGE: Save Memory ==========
    embedding = await self.embedder.embed(...)
    memory = MemoryData(...)
    memory_id = await self.storage.add(memory)

    # ========== STRATEGY 2: Graph (RelationBuilder) ==========
    await self._apply_relation_builder_strategy(memory, user_id)

    return memory_id
```

### _ProcessorDecision数据类

内部数据类，用于封装Processor的决策结果：

```python
@dataclass
class _ProcessorDecision:
    """内部决策结果"""
    action: str  # "add", "update", "delete", "none"
    memory_id: Optional[str]
    content: Optional[str]
```

### 错误处理策略

**Processor策略**：
- 失败时记录警告
- 自动降级为`add`操作
- 不影响记忆存储

**RelationBuilder策略**：
- 失败时记录警告
- 不影响记忆存储
- 关系创建失败不影响其他关系

---

## 使用示例

### 基础用法（无策略）

```python
from mempy import Memory
from mempy.embedders import OpenAIEmbedder

memory = Memory(embedder=OpenAIEmbedder())
await memory.add("我喜欢蓝色", user_id="alice")

# 所有内容无条件保存
# 不会自动创建任何关系
```

### 使用Processor策略

```python
from mempy import Memory
from mempy.core.interfaces import MemoryProcessor, ProcessorResult

class SimpleProcessor(MemoryProcessor):
    async def process(self, content, existing_memories):
        # 过滤太短的内容
        if len(content.strip()) < 5:
            return ProcessorResult(
                action="none",
                reason="内容太短"
            )

        # 检查是否有相似内容
        for mem in existing_memories:
            if content.lower() in mem.content.lower():
                return ProcessorResult(
                    action="update",
                    memory_id=mem.memory_id,
                    content=content,
                    reason="内容扩展"
                )

        # 默认添加
        return ProcessorResult(
            action="add",
            content=content,
            reason="新内容"
        )

memory = Memory(
    embedder=OpenAIEmbedder(),
    processor=SimpleProcessor(),
    verbose=True
)

await memory.add("嗨", user_id="bob")  # 会被跳过
await memory.add("我喜欢蓝色", user_id="bob")  # 会被添加
await memory.add("我真的很喜欢蓝色", user_id="bob")  # 会更新前一条
```

### 使用RelationBuilder策略

```python
from mempy import Memory
from mempy.strategies import RandomRelationBuilder, RelationType

builder = RandomRelationBuilder(
    max_relations=2,
    relation_types=[RelationType.RELATED, RelationType.SIMILAR],
    build_probability=1.0,
    seed=42
)

memory = Memory(
    embedder=OpenAIEmbedder(),
    relation_builder=builder,
    verbose=True
)

await memory.add("我喜欢蓝色", user_id="charlie")
await memory.add("蓝色是我的最爱", user_id="charlie")
await memory.add("天空是蓝色的", user_id="charlie")

# 关系会自动创建
# 无需手动调用add_relation()
```

### 使用完整策略流水线

```python
from mempy import Memory
from mempy.core.interfaces import MemoryProcessor, ProcessorResult
from mempy.strategies import RandomRelationBuilder

class SmartProcessor(MemoryProcessor):
    async def process(self, content, existing_memories):
        # 为重要内容添加元数据
        important_keywords = ["最爱", "讨厌", "重要"]
        metadata = {}

        for keyword in important_keywords:
            if keyword in content:
                metadata["importance"] = 0.9
                metadata["type"] = "preference"
                break

        return ProcessorResult(
            action="add",
            content=content,
            metadata=metadata
        )

builder = RandomRelationBuilder(max_relations=3)

memory = Memory(
    embedder=OpenAIEmbedder(),
    processor=SmartProcessor(),
    relation_builder=builder,
    verbose=True
)

await memory.add("蓝色是我的最爱", user_id="diana")
await memory.add("我喜欢蓝色", user_id="diana")
await memory.add("天空是蓝色的", user_id="diana")

# Processor: 为"最爱"添加了元数据
# Builder: 自动创建了图关系
```

---

## 最佳实践

### 1. 何时使用Processor

- **内容过滤**：过滤低质量、重复或无关内容
- **去重**：检测并合并重复记忆
- **内容增强**：为内容添加元数据、分类标签
- **智能决策**：使用LLM决定如何处理内容

### 2. 何时使用RelationBuilder

- **自动关联**：基于语义相似度自动连接相关记忆
- **时间序列**：建立时间前后的关系
- **实体关系**：基于实体重叠构建关系
- **知识图谱**：自动构建小型知识图谱

### 3. 性能考虑

- **Processor**：在每次`add()`时执行，保持轻量级
- **RelationBuilder**：在每次`add()`时执行，避免O(n²)复杂度
- **建议**：对于大规模场景，考虑批处理或异步后台处理

### 4. 测试策略

使用Mock Embedder进行单元测试：

```python
class MockEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 768

    async def embed(self, text: str) -> list:
        return [0.1] * 768

# 测试时使用mock embedder，避免API调用
memory = Memory(embedder=MockEmbedder(), processor=MyProcessor())
```

---

## 演示脚本

运行以下示例代码了解策略系统的用法：

```bash
# 策略系统完整演示
python examples/strategy_system_demo.py

# RelationBuilder演示
python examples/relation_builder_demo.py
```

---

## 相关文档

- **[RelationBuilder使用指南](relation_builder_guide.md)** - RelationBuilder详细使用教程
- **[自定义记忆演化策略](strategies.md)** - Memory evolution strategies
- **[架构设计](architecture.md)** - 核心设计原则

---

**最后更新**: 2026-01-06
**作者**: mempy开发团队
