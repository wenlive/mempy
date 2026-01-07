# mempy 架构设计文档

**面向对象**: 开发人员、架构师
**文档目的**: 理解 mempy 的核心架构和设计决策

## 目录

1. [项目概述](#1-项目概述)
2. [核心设计原则](#2-核心设计原则)
3. [架构设计](#3-架构设计)
4. [关键设计决策](#4-关键设计决策)
5. [核心接口设计](#5-核心接口设计)
6. [实现细节](#6-实现细节)

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
- Pickle 序列化持久化到 `graph.pkl`

**持久化策略**（v0.2.0+）：

图存储支持灵活的持久化策略，避免每次写操作都序列化：

```python
NetworkXGraphStore(
    persist_path,
    auto_save=False,        # 默认手动保存（破坏性变更）
    save_interval=0,        # >0 时每 N 次操作保存一次
    enable_file_lock=False  # Linux fcntl 文件锁
)
```

**使用方式**：

```python
# 方式 1: 手动控制（批量导入场景）
store = NetworkXGraphStore(path, auto_save=False)
await store.add_node(...)
await store.add_edge(...)
await store.save()  # 手动保存

# 方式 2: 定期保存（长运行服务）
store = NetworkXGraphStore(path, auto_save=True, save_interval=100)
# 每 100 次写操作自动保存一次

# 方式 3: 上下文管理器（推荐）
async with NetworkXGraphStore(path) as store:
    await store.add_node(...)
    await store.add_edge(...)
# 退出时自动保存

# 方式 4: 多进程安全（Linux only）
store = NetworkXGraphStore(
    path,
    auto_save=True,
    enable_file_lock=True  # fcntl 文件锁
)
```

**节点和边属性**：

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
- 深度遍历：BFS 手动实现（兼容旧版 NetworkX）

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

