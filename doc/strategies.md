# 自定义策略指南

## 概述

本指南介绍如何为 mempy 的记忆演化功能创建自定义策略。mempy 采用**策略模式**，允许用户替换默认的置信度演化、牢固度计算、遗忘阈值和关系探索策略。

**适用对象**：希望自定义记忆演化行为的高级用户

**前置知识**：
- 熟悉 Python 面向对象编程
- 了解抽象基类 (ABC) 和接口
- 理解策略模式

---

## 目录

- [1. 策略接口概述](#1-策略接口概述)
- [2. 自定义置信度策略](#2-自定义置信度策略)
- [3. 自定义牢固度计算](#3-自定义牢固度计算)
- [4. 自定义遗忘阈值](#4-自定义遗忘阈值)
- [5. 自定义关系探索](#5-自定义关系探索)
- [6. 完整示例](#6-完整示例)
- [7. 最佳实践](#7-最佳实践)

---

## 1. 策略接口概述

### 1.1 为什么需要策略接口？

mempy 的记忆演化功能包含多个可配置的计算逻辑：

- **置信度如何演化**：线性衰减、指数衰减、还是机器学习预测？
- **牢固度如何计算**：简单加权、复杂模型、还是神经网络？
- **何时遗忘记忆**：固定阈值、自适应、还是基于重要性？
- **如何探索关系**：余弦相似度、LLM 判断、还是图算法？

通过策略接口，你可以：
- ✅ 替换默认实现
- ✅ 适应特定场景
- ✅ 集成第三方工具
- ✅ 无需修改核心代码

### 1.2 策略接口列表

| 接口 | 用途 | 默认实现 |
|------|------|----------|
| `ConfidenceEvolutionStrategy` | 置信度演化规则 | `SimpleConfidenceStrategy` |
| `FirmnessCalculator` | 牢固度计算公式 | `WeightedFirmnessCalculator` |
| `ForgettingThresholdStrategy` | 遗忘阈值判断 | `FixedThresholdStrategy` |
| `RelationExplorationStrategy` | 关系探索算法 | `CosineSimilarityExplorer` |

### 1.3 注入自定义策略

```python
import mempy
from mempy.strategies import MyCustomStrategy

# 方式1: 初始化时注入
memory = mempy.Memory(
    embedder=my_embedder,
    confidence_strategy=MyCustomStrategy()
)

# 方式2: 方法调用时注入
await memory.explore_relations(
    explorer=MyCustomExplorer()
)
```

---

## 2. 自定义置信度策略

### 2.1 接口定义

```python
from abc import ABC, abstractmethod
from mempy.core import Memory

class ConfidenceEvolutionStrategy(ABC):
    """置信度演化策略接口"""

    @abstractmethod
    async def reinforce_on_reference(self, memory: Memory) -> float:
        """
        被其他记忆引用时的增量

        Args:
            memory: 被引用的记忆

        Returns:
            置信度增量 (通常 0.0-0.2)
        """
        pass

    @abstractmethod
    async def reinforce_on_relation(self, memory: Memory) -> float:
        """
        建立关系连接时的增量

        Args:
            memory: 建立关系的记忆

        Returns:
            置信度增量 (通常 0.0-0.1)
        """
        pass

    @abstractmethod
    async def decay_over_time(self, memory: Memory, days: int) -> float:
        """
        时间衰减量

        Args:
            memory: 要衰减的记忆
            days: 距离上次访问的天数

        Returns:
            置信度衰减量 (非负数)
        """
        pass
```

### 2.2 默认实现

```python
class SimpleConfidenceStrategy(ConfidenceEvolutionStrategy):
    """简单的增量策略"""

    async def reinforce_on_reference(self, memory: Memory) -> float:
        return 0.1

    async def reinforce_on_relation(self, memory: Memory) -> float:
        return 0.05

    async def decay_over_time(self, memory: Memory, days: int) -> float:
        # 简单的线性衰减：每天 -0.01
        return days * 0.01
```

### 2.3 自定义示例 1：指数衰减

```python
import math

class ExponentialDecayStrategy(ConfidenceEvolutionStrategy):
    """指数衰减策略"""

    async def reinforce_on_reference(self, memory: Memory) -> float:
        # 根据重要性调整增量
        base_increment = 0.1
        importance_bonus = (memory.importance - 0.5) * 0.1
        return base_increment + importance_bonus

    async def reinforce_on_relation(self, memory: Memory) -> float:
        return 0.05

    async def decay_over_time(self, memory: Memory, days: int) -> float:
        # 指数衰减公式
        base_decay = 1.0 - math.exp(-0.05 * days)

        # 访问次数多的衰减更慢
        access_bonus = min(0.3, memory.access_count * 0.01)

        return max(0.0, base_decay - access_bonus)
```

### 2.4 自定义示例 2：基于关系网络的增强

```python
class NetworkAwareConfidenceStrategy(ConfidenceEvolutionStrategy):
    """考虑关系网络的置信度策略"""

    async def reinforce_on_reference(self, memory: Memory) -> float:
        # 被高置信度记忆引用，增量更大
        # 需要通过 Memory 实例获取引用者的置信度
        # 这里简化处理，实际可扩展
        return 0.1

    async def reinforce_on_relation(self, memory: Memory) -> float:
        # 根据关系类型调整增量
        # 这里简化处理，实际可查询关系类型
        return 0.05

    async def decay_over_time(self, memory: Memory, days: int) -> float:
        # 孤立记忆衰减更快
        # 假设可以通过某种方式获取关系数量
        decay_rate = 0.01
        if hasattr(memory, '_relation_count'):
            if memory._relation_count == 0:
                decay_rate = 0.02  # 孤立记忆衰减更快
        return days * decay_rate
```

---

## 3. 自定义牢固度计算

### 3.1 接口定义

```python
from abc import ABC, abstractmethod
from mempy.core import Memory

class FirmnessCalculator(ABC):
    """牢固度计算策略接口"""

    @abstractmethod
    def calculate(
        self,
        memory: Memory,
        relation_count: int,
        avg_relation_confidence: float
    ) -> float:
        """
        计算记忆的牢固程度

        Args:
            memory: 要计算的记忆
            relation_count: 该记忆的关系数量
            avg_relation_confidence: 关联记忆的平均置信度

        Returns:
            牢固程度 (0.0-1.0)
        """
        pass
```

### 3.2 默认实现

```python
from datetime import datetime, timedelta

class WeightedFirmnessCalculator(FirmnessCalculator):
    """加权平均计算器"""

    def calculate(
        self,
        memory: Memory,
        relation_count: int,
        avg_relation_confidence: float
    ) -> float:
        # 计算最近访问得分
        recency_score = 0.0
        if memory.last_accessed_at:
            days_ago = (datetime.now() - memory.last_accessed_at).days
            recency_score = max(0.0, 1.0 - days_ago / 30.0)  # 30天内线性衰减

        # 计算访问次数得分
        count_score = min(1.0, memory.access_count / 10.0)

        # 计算关系数量得分
        relation_score = min(1.0, relation_count / 5.0)

        # 加权平均
        return (
            recency_score * 0.3 +
            count_score * 0.2 +
            relation_score * 0.2 +
            avg_relation_confidence * 0.2 +
            memory.confidence * 0.1
        )
```

### 3.3 自定义示例：ML 模型

```python
import joblib
import numpy as np

class MLBasedFirmnessCalculator(FirmnessCalculator):
    """基于机器学习的牢固度计算"""

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def calculate(
        self,
        memory: Memory,
        relation_count: int,
        avg_relation_confidence: float
    ) -> float:
        # 提取特征
        features = self._extract_features(
            memory,
            relation_count,
            avg_relation_confidence
        )

        # 预测
        firmness = self.model.predict([features])[0]
        return float(firmness)

    def _extract_features(self, memory, relation_count, avg_confidence):
        """提取特征向量"""
        # 计算距离上次访问的天数
        days_since_access = 0
        if memory.last_accessed_at:
            days_since_access = (datetime.now() - memory.last_accessed_at).days

        # 计算记忆年龄
        age_days = (datetime.now() - memory.created_at).days

        return [
            memory.confidence,
            memory.importance,
            memory.access_count,
            days_since_access,
            age_days,
            relation_count,
            avg_confidence
        ]
```

---

## 4. 自定义遗忘阈值

### 4.1 接口定义

```python
from abc import ABC, abstractmethod
from mempy.core import Memory

class ForgettingThresholdStrategy(ABC):
    """遗忘阈值策略接口"""

    @abstractmethod
    def should_forget(self, memory: Memory, firmness: float) -> bool:
        """
        判断是否应该遗忘

        Args:
            memory: 要判断的记忆
            firmness: 该记忆的牢固程度

        Returns:
            True=应该遗忘, False=保留
        """
        pass
```

### 4.2 默认实现

```python
class FixedThresholdStrategy(ForgettingThresholdStrategy):
    """固定阈值策略"""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        return firmness < self.threshold
```

### 4.3 自定义示例 1：重要性自适应

```python
class AdaptiveThresholdStrategy(ForgettingThresholdStrategy):
    """根据重要性自适应阈值"""

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        # 重要记忆更难遗忘（阈值更低）
        threshold = 0.2 + (1.0 - memory.importance) * 0.3

        # 范围: [0.2, 0.5]
        # importance=1.0 → threshold=0.2
        # importance=0.0 → threshold=0.5

        return firmness < threshold
```

### 4.4 自定义示例 2：时间感知

```python
from datetime import datetime, timedelta

class TimeAwareThresholdStrategy(ForgettingThresholdStrategy):
    """时间感知的阈值策略"""

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        age_days = (datetime.now() - memory.created_at).days

        # 新记忆更容易遗忘
        # 年龄 < 30天: threshold=0.3
        # 年龄 >= 365天: threshold=0.5
        threshold = 0.3 + min(0.2, age_days / 365 * 0.2)

        return firmness < threshold
```

### 4.5 自定义示例 3：综合策略

```python
class HybridThresholdStrategy(ForgettingThresholdStrategy):
    """综合多种因素的自适应策略"""

    def __init__(
        self,
        base_threshold: float = 0.3,
        importance_weight: float = 0.5,
        age_weight: float = 0.3,
        confidence_weight: float = 0.2
    ):
        self.base_threshold = base_threshold
        self.importance_weight = importance_weight
        self.age_weight = age_weight
        self.confidence_weight = confidence_weight

    def should_forget(self, memory: Memory, firmness: float) -> bool:
        # 基础阈值
        threshold = self.base_threshold

        # 根据重要性调整
        threshold += (1.0 - memory.importance) * self.importance_weight

        # 根据年龄调整
        age_days = (datetime.now() - memory.created_at).days
        threshold -= min(age_days / 365 * self.age_weight, threshold * 0.5)

        # 根据置信度调整
        threshold -= memory.confidence * self.confidence_weight

        # 确保阈值在合理范围
        threshold = max(0.0, min(1.0, threshold))

        return firmness < threshold
```

---

## 5. 自定义关系探索

### 5.1 接口定义

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from mempy.core import Memory, RelationType

class RelationExplorationStrategy(ABC):
    """关系探索策略接口"""

    @abstractmethod
    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        """
        探索记忆间的潜在关系

        Args:
            memories: 要探索的记忆列表
            similarity_threshold: 相似度阈值 (0.0-1.0)

        Returns:
            发现的关系列表: [(mem1, mem2, relation_type), ...]
        """
        pass
```

### 5.2 默认实现

```python
import numpy as np

class CosineSimilarityExplorer(RelationExplorationStrategy):
    """基于余弦相似度的探索器"""

    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        relations = []

        # 计算所有记忆对之间的相似度
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                similarity = self._cosine_similarity(
                    mem1.embedding,
                    mem2.embedding
                )

                if similarity >= similarity_threshold:
                    relations.append((
                        mem1,
                        mem2,
                        RelationType.SIMILAR
                    ))

        return relations

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.dot(vec1_np, vec2_np) / (
            np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
        )
```

### 5.3 自定义示例 1：LLM 引导

```python
class LLMGuidedExplorer(RelationExplorationStrategy):
    """使用 LLM 判断关系类型"""

    def __init__(self, llm_client, prompt_template: str = None):
        self.llm = llm_client
        self.prompt_template = prompt_template or self._default_prompt()

    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        relations = []

        # 只处理低置信度记忆
        low_confidence_memories = [
            m for m in memories if m.confidence < 0.7
        ]

        for i, mem1 in enumerate(low_confidence_memories):
            for mem2 in low_confidence_memories[i+1:]:
                # 使用 LLM 判断关系
                relation_type = await self._ask_llm(mem1, mem2)

                if relation_type:
                    relations.append((mem1, mem2, relation_type))

        return relations

    async def _ask_llm(
        self,
        mem1: Memory,
        mem2: Memory
    ) -> RelationType:
        """询问 LLM 两个记忆之间的关系"""
        prompt = self.prompt_template.format(
            content1=mem1.content,
            content2=mem2.content
        )

        response = await self.llm.generate(prompt)

        # 解析 LLM 响应
        if "RELATED" in response:
            return RelationType.RELATED
        elif "SIMILAR" in response:
            return RelationType.SIMILAR
        # ... 其他关系类型

        return None

    @staticmethod
    def _default_prompt():
        return """判断以下两个记忆之间的关系：

记忆1: {content1}
记忆2: {content2}

请选择关系类型（RELATED, SIMILAR, CONTRADICTORY, NONE）："""
```

### 5.4 自定义示例 2：混合策略

```python
class HybridExplorer(RelationExplorationStrategy):
    """混合相似度和 LLM 的探索器"""

    def __init__(self, llm_client, similarity_cutoff: float = 0.7):
        self.llm = llm_client
        self.similarity_cutoff = similarity_cutoff

    async def explore(
        self,
        memories: List[Memory],
        similarity_threshold: float
    ) -> List[Tuple[Memory, Memory, RelationType]]:
        relations = []

        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                # 先计算余弦相似度
                similarity = self._cosine_similarity(
                    mem1.embedding,
                    mem2.embedding
                )

                if similarity >= self.similarity_cutoff:
                    # 高相似度：直接建立关系
                    relations.append((
                        mem1,
                        mem2,
                        RelationType.SIMILAR
                    ))
                elif similarity >= similarity_threshold:
                    # 中等相似度：询问 LLM
                    relation_type = await self._ask_llm(mem1, mem2)
                    if relation_type:
                        relations.append((mem1, mem2, relation_type))

        return relations
```

---

## 6. 完整示例

### 6.1 自定义完整的记忆演化系统

```python
import math
from datetime import datetime
from mempy.strategies import (
    ConfidenceEvolutionStrategy,
    FirmnessCalculator,
    ForgettingThresholdStrategy,
    RelationExplorationStrategy
)
from mempy.core import Memory, RelationType

# 1. 自定义置信度策略
class MyConfidenceStrategy(ConfidenceEvolutionStrategy):
    async def reinforce_on_reference(self, memory: Memory) -> float:
        # 重要记忆强化更多
        base = 0.1
        bonus = (memory.importance - 0.5) * 0.1
        return base + bonus

    async def reinforce_on_relation(self, memory: Memory) -> float:
        return 0.05

    async def decay_over_time(self, memory: Memory, days: int) -> float:
        # 指数衰减
        decay = 1.0 - math.exp(-0.05 * days)
        # 访问次数多的衰减更慢
        access_bonus = min(0.3, memory.access_count * 0.01)
        return max(0.0, decay - access_bonus)

# 2. 自定义牢固度计算器
class MyFirmnessCalculator(FirmnessCalculator):
    def calculate(
        self,
        memory: Memory,
        relation_count: int,
        avg_relation_confidence: float
    ) -> float:
        # 时间衰减
        recency_score = 0.0
        if memory.last_accessed_at:
            days_ago = (datetime.now() - memory.last_accessed_at).days
            recency_score = max(0.0, 1.0 - days_ago / 30.0)

        # 非线性映射
        count_score = min(1.0, math.log(memory.access_count + 1) / 3.0)
        relation_score = min(1.0, math.sqrt(relation_count) / 3.0)

        return (
            recency_score * 0.4 +
            count_score * 0.2 +
            relation_score * 0.2 +
            avg_relation_confidence * 0.1 +
            memory.confidence * 0.1
        )

# 3. 自定义遗忘阈值
class MyForgettingStrategy(ForgettingThresholdStrategy):
    def should_forget(self, memory: Memory, firmness: float) -> bool:
        # 综合考虑重要性和置信度
        base_threshold = 0.3

        # 重要记忆更难遗忘
        threshold = base_threshold - (memory.importance - 0.5) * 0.2

        # 低置信度记忆更容易遗忘
        if memory.confidence < 0.5:
            threshold += 0.1

        return firmness < threshold

# 4. 使用所有自定义策略
import mempy

memory = mempy.Memory(
    embedder=MyEmbedder(),
    confidence_strategy=MyConfidenceStrategy(),
    firmness_calculator=MyFirmnessCalculator()
)

# 遗忘时使用自定义策略
await memory.cleanup_forgotten(
    threshold_strategy=MyForgettingStrategy()
)
```

---

## 7. 最佳实践

### 7.1 保持策略简单

**推荐**：优先使用简单的数学公式

**不推荐**：引入复杂的依赖或外部服务

**示例**：
```python
# ✅ 推荐：简单的数学公式
def decay_over_time(self, memory, days):
    return days * 0.01

# ❌ 不推荐：复杂的神经网络
def decay_over_time(self, memory, days):
    # 需要加载模型、预处理等
    ...
```

### 7.2 异步方法设计

**推荐**：所有策略方法都是异步的

**原因**：
- 与 mempy 的异步 API 一致
- 支持异步操作（如调用 LLM）
- 避免阻塞主线程

**示例**：
```python
# ✅ 正确
async def reinforce_on_reference(self, memory: Memory) -> float:
    # 可以在这里 await 异步操作
    return 0.1

# ❌ 错误
def reinforce_on_reference(self, memory: Memory) -> float:
    return 0.1
```

### 7.3 边界条件处理

**推荐**：处理边界情况

**示例**：
```python
# ✅ 正确
async def decay_over_time(self, memory: Memory, days: int) -> float:
    # 确保衰减不超过 1.0
    decay = min(days * 0.01, 1.0)
    # 确保 memory.confidence 不会变成负数
    new_confidence = max(0.0, memory.confidence - decay)
    return decay

# ❌ 错误
async def decay_over_time(self, memory: Memory, days: int) -> float:
    return days * 0.01  # 可能导致 confidence < 0
```

### 7.4 文档和测试

**推荐**：为自定义策略编写文档和测试

**示例**：
```python
class MyCustomStrategy(ConfidenceEvolutionStrategy):
    """
    自定义置信度策略

    特点：
    - 重要记忆强化更多
    - 指数衰减
    - 访问次数影响衰减速度

    公式：
    - reinforce = 0.1 + (importance - 0.5) * 0.1
    - decay = 1.0 - exp(-0.05 * days) - access_bonus
    """

    async def reinforce_on_reference(self, memory: Memory) -> float:
        # 实现...
        pass

# 测试
async def test_my_custom_strategy():
    strategy = MyCustomStrategy()
    memory = Memory(content="test", importance=0.9)

    increment = await strategy.reinforce_on_reference(memory)
    assert increment == 0.15  # 0.1 + (0.9 - 0.5) * 0.1
```

### 7.5 性能考虑

**推荐**：避免在策略中进行重复的昂贵计算

**示例**：
```python
# ✅ 正确：缓存计算结果
class CachedCalculator(FirmnessCalculator):
    def __init__(self):
        self._cache = {}

    def calculate(self, memory, relation_count, avg_confidence):
        cache_key = (memory.memory_id, relation_count, avg_confidence)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._calculate_impl(memory, relation_count, avg_confidence)
        self._cache[cache_key] = result
        return result

# ❌ 错误：每次都重新计算
class ExpensiveCalculator(FirmnessCalculator):
    def calculate(self, memory, relation_count, avg_confidence):
        # 每次都调用外部 API
        return self._call_external_api(memory)
```

### 7.6 错误处理

**推荐**：妥善处理异常

**示例**：
```python
# ✅ 正确
async def reinforce_on_reference(self, memory: Memory) -> float:
    try:
        # 尝试获取额外信息
        bonus = await self._calculate_bonus(memory)
        return 0.1 + bonus
    except Exception as e:
        # 降级到默认值
        logger.warning(f"Failed to calculate bonus: {e}")
        return 0.1

# ❌ 错误：异常会中断整个流程
async def reinforce_on_reference(self, memory: Memory) -> float:
    bonus = await self._calculate_bonus(memory)  # 可能抛出异常
    return 0.1 + bonus
```

---

## 附录

### A. 常见问题

**Q: 可以同时使用多个策略吗？**

A: 不可以。每个策略类型只能使用一个实现。但你可以创建组合策略：

```python
class CombinedConfidenceStrategy(ConfidenceEvolutionStrategy):
    def __init__(self, strategy1, strategy2):
        self.s1 = strategy1
        self.s2 = strategy2

    async def reinforce_on_reference(self, memory):
        # 组合两个策略的结果
        inc1 = await self.s1.reinforce_on_reference(memory)
        inc2 = await self.s2.reinforce_on_reference(memory)
        return (inc1 + inc2) / 2
```

**Q: 策略可以访问全局状态吗？**

A: 可以，但不推荐。策略应该是无状态的，或者只依赖自己的配置。如果需要共享状态，使用类属性或依赖注入。

**Q: 如何调试自定义策略？**

A: 使用日志记录和单元测试：

```python
import logging

class DebugStrategy(ConfidenceEvolutionStrategy):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def reinforce_on_reference(self, memory):
        increment = 0.1
        self.logger.debug(
            f"Reinforcing memory {memory.memory_id} "
            f"by {increment} (confidence: {memory.confidence})"
        )
        return increment
```

### B. 参考资料

- [策略模式 - Refactoring Guru](https://refactoring.guru/design-patterns/strategy-pattern)
- [ABC 抽象基类 - Python 文档](https://docs.python.org/3/library/abc.html)
- [mempy API 文档](api.md)
- [记忆演化功能设计](evolution.md)

---

**文档结束**
