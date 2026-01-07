# RelationBuilder 使用指南

## 简介

RelationBuilder是mempy策略系统的一部分，用于在添加新记忆时自动构建图关系。与RelationExplorationStrategy不同，RelationBuilder是**实时、增量**地构建关系。

## 与RelationExplorationStrategy的区别

| 维度 | RelationBuilder | RelationExplorationStrategy |
|------|----------------|----------------------------|
| **使用场景** | 添加新记忆时自动构建 | 批量探索现有记忆关系 |
| **调用时机** | `Memory.add()`时 | `Memory.evolve()`时 |
| **执行方式** | 实时、增量 | 批量、事后 |
| **性能影响** | 每次add()时执行 | 定期批量执行 |
| **适用规模** | 中小规模 | 大规模知识图谱 |

**选择建议**：
- 如果需要在**添加记忆时立即建立关系**，使用RelationBuilder
- 如果需要**批量探索和发现隐藏关系**，使用RelationExplorationStrategy

---

## 基础用法

### 1. 使用RandomRelationBuilder

```python
from mempy import Memory
from mempy.strategies import RandomRelationBuilder, RelationType

builder = RandomRelationBuilder(
    max_relations=2,  # 每个新记忆最多创建2个关系
    relation_types=[RelationType.RELATED, RelationType.SIMILAR],
    build_probability=1.0,  # 100%概率构建关系
    seed=42  # 随机种子
)

memory = Memory(
    embedder=MyEmbedder(),
    relation_builder=builder
)

await memory.add("我喜欢蓝色", user_id="alice")
await memory.add("蓝色是我的最爱", user_id="alice")

# 关系会自动创建！
```

### 2. 不使用Builder（默认）

```python
memory = Memory(embedder=MyEmbedder())

await memory.add("我喜欢蓝色", user_id="alice")
await memory.add("蓝色是我的最爱", user_id="alice")

# 不会自动创建关系
# 保持向后兼容性
```

---

## 参数说明

### RandomRelationBuilder参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_relations` | int | 3 | 每个新记忆最多创建的关系数 |
| `relation_types` | List[RelationType] | [RELATED] | 随机选择的关系类型列表 |
| `build_probability` | float | 1.0 | 构建关系的概率（0.0-1.0） |
| `seed` | Optional[int] | None | 随机种子，用于可重现性 |

### 参数使用示例

```python
# 每次最多创建5个关系
builder = RandomRelationBuilder(max_relations=5)

# 只RELATED和SIMILAR两种类型
builder = RandomRelationBuilder(
    relation_types=[RelationType.RELATED, RelationType.SIMILAR]
)

# 只在50%的情况下构建关系（用于测试）
builder = RandomRelationBuilder(build_probability=0.5)

# 固定随机种子以获得可重现结果
builder = RandomRelationBuilder(seed=123)
```

---

## 自定义RelationBuilder

### 示例1: 语义相似度构建器

基于向量相似度自动连接语义相近的记忆。

```python
from mempy.strategies import RelationBuilder
from mempy.core.memory import Memory, RelationType
import numpy as np

class SemanticRelationBuilder(RelationBuilder):
    """基于语义相似度构建关系"""

    def __init__(self, threshold: float = 0.95):
        """
        Args:
            threshold: 相似度阈值（0-1），超过此值才创建关系
        """
        self.threshold = threshold

    async def build(
        self,
        new_memory: Memory,
        existing_memories: list
    ) -> list:
        relations = []
        new_vec = np.array(new_memory.embedding)

        for existing in existing_memories:
            existing_vec = np.array(existing.embedding)

            # 计算余弦相似度
            similarity = np.dot(new_vec, existing_vec) / (
                np.linalg.norm(new_vec) * np.linalg.norm(existing_vec)
            )

            if similarity >= self.threshold:
                relations.append((
                    new_memory.memory_id,
                    existing.memory_id,
                    RelationType.SIMILAR,
                    {
                        "similarity": float(similarity),
                        "threshold": self.threshold
                    }
                ))

        return relations

# 使用
builder = SemanticRelationBuilder(threshold=0.98)
memory = Memory(embedder=MyEmbedder(), relation_builder=builder)
await memory.add("我喜欢蓝色", user_id="bob")
```

### 示例2: 时间序列关系构建器

基于时间前后关系自动构建FOLLOWS关系。

```python
from mempy.strategies import RelationBuilder
from mempy.core.memory import Memory, RelationType
from datetime import datetime

class TemporalRelationBuilder(RelationBuilder):
    """基于时间序列构建关系"""

    def __init__(self, time_window_seconds: int = 3600):
        """
        Args:
            time_window_seconds: 时间窗口（秒），默认1小时
        """
        self.time_window = time_window_seconds

    async def build(
        self,
        new_memory: Memory,
        existing_memories: list
    ) -> list:
        relations = []
        new_time = new_memory.created_at

        for existing in existing_memories:
            if not existing.created_at:
                continue

            time_diff = (new_time - existing.created_at).total_seconds()

            # 如果新记忆在时间窗口内创建，建立FOLLOWS关系
            if 0 < time_diff <= self.time_window:
                relations.append((
                    new_memory.memory_id,
                    existing.memory_id,
                    RelationType.FOLLOWS,
                    {
                        "time_diff_seconds": time_diff,
                        "time_window": self.time_window
                    }
                ))

        return relations

# 使用
builder = TemporalRelationBuilder(time_window_seconds=3600)
memory = Memory(embedder=MyEmbedder(), relation_builder=builder)
```

### 示例3: 实体重叠关系构建器

基于实体重叠度构建关系（需要实体提取器）。

```python
from mempy.strategies import RelationBuilder
from mempy.core.memory import Memory, RelationType

class EntityOverlapRelationBuilder(RelationBuilder):
    """基于实体重叠构建关系"""

    def __init__(self, entity_extractor, min_overlap: int = 2):
        """
        Args:
            entity_extractor: 实体提取器对象
            min_overlap: 最小重叠实体数
        """
        self.entity_extractor = entity_extractor
        self.min_overlap = min_overlap

    async def build(
        self,
        new_memory: Memory,
        existing_memories: list
    ) -> list:
        relations = []

        # 提取新记忆的实体
        new_entities = set(self.entity_extractor.extract(new_memory.content))

        for existing in existing_memories:
            # 提取已存在记忆的实体
            existing_entities = set(
                self.entity_extractor.extract(existing.content)
            )

            # 计算重叠实体
            overlap = new_entities & existing_entities

            if len(overlap) >= self.min_overlap:
                relations.append((
                    new_memory.memory_id,
                    existing.memory_id,
                    RelationType.RELATED,
                    {
                        "shared_entities": list(overlap),
                        "overlap_count": len(overlap)
                    }
                ))

        return relations

# 使用（需要实体提取器）
class SimpleEntityExtractor:
    def extract(self, text: str) -> list:
        # 简单实现：提取大写单词作为实体
        import re
        return list(set(re.findall(r'\b[A-Z][a-z]+\b', text)))

builder = EntityOverlapRelationBuilder(
    entity_extractor=SimpleEntityExtractor(),
    min_overlap=2
)
memory = Memory(embedder=MyEmbedder(), relation_builder=builder)
```

---

## 高级用法

### 组合多个RelationBuilder

```python
from mempy.strategies import RelationBuilder

class MultiRelationBuilder(RelationBuilder):
    """组合多个构建器"""

    def __init__(self, builders: list):
        self.builders = builders

    async def build(self, new_memory, existing_memories):
        all_relations = []

        for builder in self.builders:
            relations = await builder.build(new_memory, existing_memories)
            all_relations.extend(relations)

        return all_relations

# 组合多个构建器
builder = MultiRelationBuilder([
    SemanticRelationBuilder(threshold=0.95),
    TemporalRelationBuilder(time_window_seconds=3600)
])

memory = Memory(embedder=MyEmbedder(), relation_builder=builder)
```

### 条件性构建

```python
class ConditionalRelationBuilder(RelationBuilder):
    """根据条件决定是否构建关系"""

    def __init__(self, builder, condition_fn):
        self.builder = builder
        self.condition_fn = condition_fn

    async def build(self, new_memory, existing_memories):
        # 检查条件
        if not self.condition_fn(new_memory):
            return []

        # 条件满足，执行构建
        return await self.builder.build(new_memory, existing_memories)

# 只为重要内容构建关系
def is_important(memory):
    return memory.metadata.get("importance", 0) > 0.8

base_builder = SemanticRelationBuilder(threshold=0.9)
builder = ConditionalRelationBuilder(base_builder, is_important)

memory = Memory(
    embedder=MyEmbedder(),
    relation_builder=builder
)
```

---

## 最佳实践

### 1. 选择合适的构建粒度

- **过于密集**：每条记忆都连接到很多其他记忆，图变得混乱
- **过于稀疏**：关系太少，失去图的意义
- **建议**：每条记忆2-5个关系是比较合适的范围

### 2. 使用metadata存储关系上下文

```python
relations.append((
    new_id,
    existing_id,
    RelationType.SIMILAR,
    {
        "similarity": 0.95,
        "method": "cosine",
        "threshold": 0.9,
        "model": "text-embedding-ada-002"
    }
))
```

### 3. 考虑性能

RelationBuilder在每次`add()`时执行，需要考虑：

- **避免O(n²)**：不要与所有现有记忆比较
- **限制搜索范围**：使用`limit`参数限制候选记忆数
- **使用缓存**：缓存昂贵的计算结果

```python
async def build(self, new_memory, existing_memories):
    # 只考虑最近的100条记忆
    candidates = existing_memories[:100]

    # 或使用search()预先筛选
    candidates = await memory.search(
        new_memory.content,
        user_id=new_memory.user_id,
        limit=20
    )

    # 在候选中构建关系
    ...
```

### 4. 测试自定义构建器

使用Mock Embedder测试自定义构建器：

```python
import pytest
from mempy.strategies import RelationBuilder
from mempy import Memory, Embedder

class MockEmbedder(Embedder):
    @property
    def dimension(self):
        return 128

    async def embed(self, text):
        return [0.5] * 128

@pytest.mark.asyncio
async def test_my_builder():
    builder = MyRelationBuilder()
    memory = Memory(embedder=MockEmbedder(), relation_builder=builder)

    await memory.add("记忆1", user_id="test")
    await memory.add("记忆2", user_id="test")

    # 验证关系是否创建
    all_memories = await memory.get_all(user_id="test")
    for mem in all_memories:
        relations = await memory.get_relations(mem.memory_id)
        assert len(relations) > 0  # 根据你的builder调整断言
```

---

## 故障排查

### 问题1: 关系没有创建

**可能原因**：
- `build_probability`太低
- `existing_memories`为空（第一批记忆）
- 你的builder逻辑没有返回relations

**调试方法**：
```python
class MyBuilder(RelationBuilder):
    async def build(self, new_memory, existing_memories):
        print(f"New memory: {new_memory.memory_id}")
        print(f"Existing memories: {len(existing_memories)}")

        relations = []
        # 你的逻辑

        print(f"Created {len(relations)} relations")
        return relations
```

### 问题2: 性能太慢

**优化方法**：
1. 限制候选记忆数量
2. 使用近似算法（如LSH）
3. 降低`max_relations`值
4. 降低`build_probability`值

### 问题3: 循环导入

**解决方法**：使用字符串前向引用

```python
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from mempy.core.memory import Memory

class MyBuilder(RelationBuilder):
    async def build(
        self,
        new_memory: "Memory",
        existing_memories: List["Memory"]
    ):
        ...
```

---

## 演示脚本

运行RelationBuilder演示：

```bash
python examples/relation_builder_demo.py
```

该演示包含：
- RandomRelationBuilder基础用法
- 自定义SemanticRelationBuilder
- 参数配置示例

---

## 相关文档

- **[策略系统架构](strategy_system.md)** - 策略系统总体设计
- **[自定义记忆演化策略](strategies.md)** - RelationExplorationStrategy
- **[架构设计](architecture.md)** - 核心设计原则

---

**最后更新**: 2026-01-06
**作者**: mempy开发团队
