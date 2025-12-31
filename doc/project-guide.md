# mempy 项目说明文档

## 目录

1. [项目概述](#1-项目概述)
2. [核心设计原则](#2-核心设计原则)
3. [架构设计](#3-架构设计)
4. [关键设计决策](#4-关键设计决策)
5. [核心接口设计](#5-核心接口设计)
6. [实现细节](#6-实现细节)
7. [使用指南](#7-使用指南)
8. [扩展指南](#8-扩展指南)
9. [待办与未来方向](#9-待办与未来方向)

---

## 1. 项目概述

### 1.1 项目定位

mempy 是一个类似 [mem0](https://github.com/mem0ai/mem0) 的记忆管理库，核心特性：

- **零配置嵌入式**：pip install + import 即用
- **向量+图双一等公民**：语义搜索 + 关系推理
- **可插拔设计**：嵌入器、处理器、存储后端均可扩展
- **全异步 API**：原生 async/await 支持

### 1.2 设计愿景

最终愿景是构建一个**原生的嵌入式向量-图数据库**，同时支持：
- 向量存储（语义搜索）
- 图存储（关系推理）
- 时序数据（可选）
- KV 存储（可选）

当前 MVP 阶段采用**双写方案**（ChromaDB + NetworkX）作为技术过渡，通过统一抽象层屏蔽底层复杂性，为未来替换专用数据库铺平道路。

### 1.3 与 mem0 的对比

| 特性 | mem0 | mempy |
|------|------|-------|
| 向量存储 | 多种后端可选 | ChromaDB（简化） |
| 图存储 | Neptune（需外部服务） | NetworkX（嵌入式） |
| LLM 依赖 | 内置 OpenAI 默认 | 用户自实现 |
| 部署方式 | 云服务 + 自托管 | 纯嵌入式 |
| API 风格 | 混合同步/异步 | 全异步 |

---

## 2. 核心设计原则

### 2.1 用户无感知

隐藏所有底层复杂性，用户只需：
1. 实现 `Embedder` 接口（声明维度 + 嵌入方法）
2. 调用 `Memory` API

### 2.2 渐进式复杂度

```
简单用法           高级用法
    ↓                 ↓
Memory.add()  →  自定义 Processor
              →  自定义 Storage Backend
              →  完全定制
```

### 2.3 向量为主，图为辅助

- **向量存储是主存储**：失败 = 整体失败
- **图存储是辅助存储**：失败 = 警告，不影响主流程

这保证了核心功能（语义搜索）的可靠性。

### 2.4 可观测性优先

所有关键操作都有日志输出：
- `[VECTOR]` 前缀：向量存储操作
- `[GRAPH]` 前缀：图存储操作
- `[PROCESSOR]` 前缀：处理器操作
- `✓` / `✗` / `⚠` 符号：成功/失败/警告

---

## 3. 架构设计

### 3.1 三层架构

```
┌─────────────────────────────────────────────┐
│           应用层 (Application)              │
│  • Memory (主 API)                          │
│  • 用户代码                                  │
├─────────────────────────────────────────────┤
│           服务层 (Service)                  │
│  • MemoryProcessor (智能处理)               │
│  • LLMProcessor (LLM 实现)                  │
├─────────────────────────────────────────────┤
│           存储层 (Storage)                  │
│  • DualStorageBackend (统一抽象)            │
│  ├─ ChromaVectorStore (向量)               │
│  └─ NetworkXGraphStore (图)                │
└─────────────────────────────────────────────┘
```

### 3.2 目录结构

```
mempy/
├── __init__.py           # 公共 API 导出
├── memory.py             # Memory 主类（用户入口）
├── config.py             # 配置管理
├── core/                 # 核心数据结构和接口
│   ├── memory.py         # Memory, RelationType, Relation
│   ├── interfaces.py     # Embedder, MemoryProcessor, StorageBackend
│   └── exceptions.py     # 异常定义
├── storage/              # 存储实现
│   ├── backend.py        # DualStorageBackend
│   ├── vector_store.py   # ChromaDB 向量存储
│   └── graph_store.py    # NetworkX 图存储
└── processors/           # 处理器实现
    ├── base.py           # MemoryProcessor 抽象类
    └── llm_processor.py  # LLMProcessor 实现
```

### 3.3 数据流

```
用户调用 Memory.add()
        ↓
    (可选) Processor.process()
        ↓
    判断操作类型: add/update/delete/none
        ↓
    Embedder.embed() 生成向量
        ↓
    DualStorageBackend.add()
        ↓
    ┌───────────────────────┐
    │ 向量存储 (ChromaDB)    │ ← 主存储
    │ 图存储 (NetworkX)      │ ← 辅助存储
    └───────────────────────┘
```

---

## 4. 关键设计决策

### 4.1 嵌入器接口：用户必须声明维度

**问题**：ChromaDB 需要知道向量维度才能创建集合。

**解决方案**：用户在 `Embedder` 中必须实现 `dimension` 属性。

```python
class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:  # 用户必须实现
        """返回向量维度"""
        pass
```

**理由**：
- 让用户明确知道自己模型的维度
- 避免首次调用时的隐式推断错误
- 接口更清晰

### 4.2 存储路径：三级优先级

**优先级**：用户传入 > 环境变量 > 默认值

```python
def get_storage_path(user_path: Optional[str] = None) -> Path:
    if user_path:
        return Path(user_path)
    env_path = os.environ.get("MEMPY_HOME")
    if env_path:
        return Path(env_path)
    return Path.home() / ".mempy" / "data"
```

**使用方式**：
```python
# 方式1：默认 ~/.mempy/data
memory = Memory(embedder=embedder)

# 方式2：环境变量
# export MEMPY_HOME=/custom/path
memory = Memory(embedder=embedder)

# 方式3：直接传入
memory = Memory(embedder=embedder, storage_path="/my/path")
```

### 4.3 关系类型：基于关系代数的枚举

**设计思路**：参考关系代数的核心运算

| 关系代数运算 | 对应关系类型 | 语义 |
|-------------|-------------|------|
| Selection/Join | `RELATED` | 相关系数 |
| Union | `EQUIVALENT` | 等价可合并 |
| Difference | `CONTRADICTORY` | 矛盾互斥 |
| 笛卡尔积分解 | `PART_OF` | 组成关系 |
| - | `GENERALIZATION` / `SPECIALIZATION` | 泛化/特化 |

**完整枚举**：
```python
class RelationType(Enum):
    # 基础关系
    RELATED         # 相关系数
    EQUIVALENT      # 等价关系
    CONTRADICTORY   # 矛盾关系

    # 层次关系
    GENERALIZATION  # 泛化（超类）
    SPECIALIZATION  # 特化（子类）
    PART_OF         # 组成关系

    # 时序/因果
    PRECEDES        # 先于
    FOLLOWS         # 后于
    CAUSES          # 导致
    CAUSED_BY       # 被导致

    # 语义关联
    SIMILAR         # 相似
    PROPERTY_OF     # 属性关系
    INSTANCE_OF     # 实例关系
    CONTEXT_FOR     # 上下文关系
```

### 4.4 MemoryProcessor Prompt：简化设计

**决策**：不采用 mem0 的复杂格式（多个记忆的 ADD/UPDATE/DELETE），改为单个操作判断。

**Prompt 模板**：
```python
PROCESSOR_PROMPT = """你是一个记忆管理助手。根据用户输入和已有记忆，判断需要执行的操作。

已有记忆：
{existing_memories}

用户输入：{user_input}

请返回 JSON 格式：
{
  "action": "add" | "update" | "delete" | "none",
  "memory_id": "...",      // update 或 delete 时填写
  "content": "...",        // update 时填写新内容
  "reason": "..."          // 判断理由
}
"""
```

**理由**：
- 更简单，降低 LLM 出错概率
- MVP 阶段够用
- 复杂场景可由用户扩展

### 4.5 双写一致性策略

**策略**：向量为主，图为辅

```
┌─────────────────────────────────────────┐
│  add(memory)                             │
├─────────────────────────────────────────┤
│  1. 写向量存储                           │
│     ├─ 成功 → 继续                       │
│     └─ 失败 → 整体失败 ✗                 │
│  2. 写图存储                             │
│     ├─ 成功 → 完成 ✓                     │
│     └─ 失败 → 记录警告，但仍成功 ⚠        │
└─────────────────────────────────────────┘
```

**代码体现**：
```python
async def add(self, memory: Memory) -> str:
    # 1. 向量存储（主）
    try:
        memory_id = await self.vector_store.add(memory)
        self.logger.info(f"[VECTOR] ✓ Added memory {memory_id}")
    except Exception as e:
        self.logger.error(f"[VECTOR] ✗ Failed: {e}")
        raise StorageError(...)  # 主存储失败 = 整体失败

    # 2. 图存储（辅）
    try:
        await self.graph_store.add_node(memory_id, memory)
        self.logger.info(f"[GRAPH] ✓ Added node {memory_id}")
    except Exception as e:
        # 图失败不影响整体
        self.logger.warning(f"[GRAPH] ⚠ Failed (non-critical): {e}")

    return memory_id
```

### 4.6 可观测性设计

**三层日志**：

1. **storage 层**：记录底层操作
   ```python
   self.logger.info(f"[VECTOR] ✓ Added memory {memory_id}")
   self.logger.warning(f"[GRAPH] ⚠ Failed (non-critical): {e}")
   ```

2. **Memory 层**：记录用户操作
   ```python
   if self.verbose:
       self.logger.info(f"Processing: {content[:50]}...")
       self.logger.info(f"Saved: {memory_id}")
   ```

3. **返回值**：`OperationResult` 记录详细状态
   ```python
   @dataclass
   class OperationResult:
       success: bool
       operation: str
       vector_store_ok: Optional[bool]
       graph_store_ok: Optional[bool]
       error_message: Optional[str]
       timestamp: datetime
   ```

---

## 5. 核心接口设计

### 5.1 Embedder 接口

用户必须实现的嵌入器接口：

```python
class Embedder(ABC):
    """嵌入器抽象接口"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度（必须实现）"""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """生成文本的嵌入向量"""
        pass
```

**实现示例**：
```python
class MyEmbedder(Embedder):
    def __init__(self, endpoint: str, dimension: int):
        self.endpoint = endpoint
        self._dimension = dimension

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

### 5.2 MemoryProcessor 接口

可选的智能处理器接口：

```python
class MemoryProcessor(ABC):
    """记忆处理器抽象接口"""

    @abstractmethod
    async def process(
        self,
        content: str,
        existing_memories: List[Memory]
    ) -> ProcessorResult:
        """
        判断操作类型

        Returns:
            ProcessorResult(
                action="add" | "update" | "delete" | "none",
                memory_id=...,  # update/delete 时
                content=...,    # update 时
                reason=...
            )
        """
        pass
```

### 5.3 StorageBackend 接口

统一存储抽象，未来可用于替换专用数据库：

```python
class StorageBackend(ABC):
    """统一存储接口"""

    @abstractmethod
    async def add(self, memory: Memory) -> str: ...

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]: ...

    @abstractmethod
    async def search(self, query_vector, filters, limit) -> List[Memory]: ...

    @abstractmethod
    async def update(self, memory_id: str, memory: Memory): ...

    @abstractmethod
    async def delete(self, memory_id: str): ...

    @abstractmethod
    async def add_relation(self, from_id, to_id, relation_type, metadata): ...

    @abstractmethod
    async def get_relations(self, memory_id, direction, max_depth): ...
```

---

## 6. 实现细节

### 6.1 Memory 数据类

```python
@dataclass
class Memory:
    memory_id: str
    content: str
    embedding: List[float]
    user_id: Optional[str] = None      # 用户维度隔离
    agent_id: Optional[str] = None      # Agent 维度隔离
    run_id: Optional[str] = None        # 会话维度隔离
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    priority: float = 0.5               # 重要性（未来用于排序）
    confidence: float = 1.0             # 置信度（未来用于过滤）
```

### 6.2 ChromaDB 向量存储

**关键设计**：
- 使用 `cosine` 相似度
- 元数据存储：content, user_id, agent_id, run_id, priority, confidence
- 额外 metadata 序列化为 JSON 存储

```python
def _memory_to_metadata(self, memory: Memory) -> Dict[str, Any]:
    return {
        "content": memory.content,
        "user_id": memory.user_id or "",
        "agent_id": memory.agent_id or "",
        "run_id": memory.run_id or "",
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else "",
        "priority": memory.priority,
        "confidence": memory.confidence,
        "metadata_json": json.dumps(memory.metadata),
    }
```

### 6.3 NetworkX 图存储

**关键设计**：
- 使用 `DiGraph` 有向图
- 节点存储 Memory 的完整属性（不含 embedding）
- 边存储 relation_type 和 metadata

```python
# 节点属性
attrs = {
    "memory_id": memory.memory_id,
    "content": memory.content,
    "user_id": memory.user_id,
    ...
}

# 边属性
edge_attrs = {
    "type": relation_type.value,
    "metadata": json.dumps(metadata or {}),
}
```

**关系查询**：
- 直接邻居：`graph.successors(id)` / `graph.predecessors(id)`
- 路径查找：`nx.shortest_path(graph, from, to)`
- 广度优先：`nx.descendants_at_distance()`

### 6.4 双写实现

`DualStorageBackend` 作为统一入口：

```python
class DualStorageBackend(StorageBackend):
    def __init__(self, persist_path: Path):
        self.vector_store = ChromaVectorStore(persist_path)
        self.graph_store = NetworkXGraphStore(persist_path)
        self._setup_logging()

    async def add(self, memory: Memory) -> str:
        # 1. 先写向量（主）
        memory_id = await self.vector_store.add(memory)
        # 2. 再写图（辅）
        await self.graph_store.add_node(memory_id, memory)
        return memory_id
```

---

## 7. 使用指南

### 7.1 基础用法

```python
import mempy

# 1. 实现 Embedder
class MyEmbedder(mempy.Embedder):
    def __init__(self):
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        return await my_llm.embed(text)

# 2. 创建 Memory 实例
memory = mempy.Memory(embedder=MyEmbedder(), verbose=True)

# 3. 添加记忆
mem1 = await memory.add("I like blue", user_id="alice")
mem2 = await memory.add("Alice works at Google", user_id="alice")

# 4. 搜索
results = await memory.search("Alice's job", user_id="alice")

# 5. 添加关系
await memory.add_relation(mem1, mem2, mempy.RelationType.PROPERTY_OF)

# 6. 获取关系
relations = await memory.get_relations(mem1, max_depth=2)
```

### 7.2 使用 LLM 处理器

```python
from mempy.processors import LLMProcessor

async def my_llm_call(prompt: str) -> str:
    # 调用你的 LLM
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/generate",
            json={"prompt": prompt}
        ) as resp:
            return await resp.text()

processor = LLMProcessor(llm_call=my_llm_call)
memory = mempy.Memory(embedder=embedder, processor=processor)
```

### 7.3 配置存储路径

```bash
# 环境变量
export MEMPY_HOME=/custom/path
```

```python
# 或直接传入
memory = mempy.Memory(
    embedder=embedder,
    storage_path="/my/custom/path"
)
```

---

## 8. 扩展指南

### 8.1 自定义 Embedder

支持任何能生成向量的服务：

```python
class OpenAIEmbedder(mempy.Embedder):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._dimension = 1536  # text-embedding-ada-002

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        # 调用 OpenAI API
        ...
```

### 8.2 自定义 Processor

实现自定义的决策逻辑：

```python
class RuleBasedProcessor(mempy.MemoryProcessor):
    async def process(self, content: str, existing_memories: List[Memory]):
        # 简单规则：如果内容相似度 > 0.9，则 update
        for m in existing_memories:
            if similarity(content, m.content) > 0.9:
                return mempy.ProcessorResult(
                    action="update",
                    memory_id=m.memory_id,
                    content=content,
                    reason="High similarity"
                )
        return mempy.ProcessorResult(action="add", reason="New content")
```

### 8.3 自定义 Storage Backend

未来替换为专用数据库时，只需实现 `StorageBackend` 接口：

```python
class MyUnifiedDatabase(StorageBackend):
    """未来的专用向量-图数据库"""

    async def add(self, memory: Memory) -> str:
        # 单一存储，同时支持向量和图
        return self.db.insert(memory)

    async def search(self, query_vector, filters, limit):
        # 混合搜索
        return self.db.hybrid_search(query_vector, filters, limit)

    ...
```

然后：
```python
memory = mempy.Memory(
    embedder=embedder,
    storage_backend=MyUnifiedDatabase(...)  # 未来支持
)
```

---

## 9. 待办与未来方向

### 9.1 当前限制

| 限制 | 说明 |
|------|------|
| 单机限制 | 仅支持单机部署，无分布式 |
| 内存限制 | 大规模记忆需要足够内存 |
| 并发限制 | 写操作需要锁机制（待实现） |

### 9.2 短期计划

- [ ] 添加 history 功能（记录记忆变更历史）
- [ ] 添加单元测试覆盖
- [ ] 性能优化（批量操作、缓存）
- [ ] 添加更多 embedder 示例

### 9.3 中期计划

- [ ] 支持更多向量存储后端
- [ ] 添加 Web 管理界面
- [ ] 实现 Memory 导出/导入
- [ ] 添加记忆合并功能

### 9.4 长期愿景

- [ ] **实现专用的嵌入式向量-图数据库**
- [ ] 支持时序数据和 KV 存储
- [ ] 分布式版本
- [ ] 移动端支持

---

## 附录

### A. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1.0 | 2024-12 | MVP 版本，核心功能实现 |

### B. 参考资料

- [mem0 GitHub](https://github.com/mem0ai/mem0)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [NetworkX 文档](https://networkx.org/)

### C. 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 发起 Pull Request
