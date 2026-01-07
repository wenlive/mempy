# mempy测试指南

## 测试概览

memppy使用pytest进行测试，包含以下类型：
- **单元测试**: 测试单个类/函数
- **集成测试**: 测试模块间交互
- **Benchmark测试**: 性能基准评估
- **验证脚本**: 开发时功能验证

## 快速开始

```bash
# 安装测试依赖
pip install -e ".[test]"

# 运行所有测试
pytest tests/

# 运行所有单元测试
pytest tests/unit/ -v

# 运行所有集成测试
pytest tests/integration/ -v

# 按标记运行测试
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "not api"  # 排除需要API的测试
pytest tests/ -m "not slow"  # 排除慢速测试

# 运行特定测试文件
pytest tests/unit/test_core.py -v

# 查看可用标记
pytest --markers
```

## API密钥配置

某些测试需要API密钥（用于benchmark测试）：

```bash
# 复制示例配置
cp tests/benchmarks/.env.example tests/benchmarks/.env

# 编辑.env文件，添加你的API密钥
# ZHIPUAI_API_KEY=your-key-here
```

⚠️ **安全提示**：
- 永远不要提交真实API密钥到仓库
- `.env`文件已在`.gitignore`中
- 只提交`.env.example`作为模板

## 测试目录说明

### 目录结构

```
tests/
├── conftest.py                   # pytest配置和fixtures
├── pytest.ini                    # pytest配置文件
│
├── unit/                         # 单元测试（快速、独立）
│   ├── __init__.py
│   ├── test_config.py            # 配置管理测试
│   ├── test_core.py              # 核心数据类测试
│   ├── test_strategies.py        # 策略实现测试
│   └── test_processors.py        # 处理器测试
│
├── integration/                  # 集成测试（模块交互）
│   ├── __init__.py
│   ├── test_builders.py          # 关系构建器测试
│   ├── test_evolution.py         # 演化功能测试
│   ├── test_strategy_system.py   # 策略系统集成测试
│   ├── test_graph_store.py       # 图存储测试
│   ├── test_storage_backend.py   # 存储后端测试
│   └── test_memory_api.py        # Memory API测试
│
├── benchmarks/                   # 性能基准测试
│   ├── adapters/                 # 模型适配器
│   │   ├── mock.py               # Mock适配器
│   │   └── zhipu.py              # 智谱AI适配器
│   │
│   ├── locomo/                   # LOCOMO数据集评估
│   │   ├── dataset.py            # 数据集类
│   │   ├── evaluator.py          # 评估器实现
│   │   ├── test_e2e_zhipu.py     # 智谱AI端到端测试（支持多种模式）
│   │   └── data/
│   │       └── locomo10.json     # 测试数据集
│   │
│   └── verification/             # 功能验证脚本（非pytest测试）
│       ├── __init__.py
│       ├── compare_embedders.py  # Embedder性能对比
│       ├── compare_relation_builders.py  # 关系构建器效果对比
│       └── test_relation_builder_simple.py  # RelationBuilder功能验证
│
└── data/                         # 测试数据
```

### 单元测试 (tests/unit/)

快速、独立的测试，不依赖外部服务或复杂设置。

| 文件 | 测试内容 | Mock? | API? |
|------|---------|-------|------|
| test_config.py | 配置管理 | ✓ | ✗ |
| test_core.py | 核心数据类 | ✓ | ✗ |
| test_strategies.py | 策略实现 | ✓ | ✗ |
| test_processors.py | 处理器逻辑 | ✓ | ✗ |

**运行单元测试**:
```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行特定单元测试
pytest tests/unit/test_core.py -v

# 使用标记运行
pytest tests/ -m "unit"
```

### 集成测试 (tests/integration/)

测试多个模块协同工作，可能涉及文件系统和存储操作。

| 文件 | 测试内容 | Mock? | API? |
|------|---------|-------|------|
| test_builders.py | 关系构建器 | 部分 | ✗ |
| test_evolution.py | 演化功能 | 部分 | ✗ |
| test_strategy_system.py | 策略系统 | 部分 | ✗ |
| test_graph_store.py | 图存储 | 部分 | ✗ |
| test_storage_backend.py | 存储后端 | 部分 | ✗ |
| test_memory_api.py | Memory API | 部分 | ✗ |

**运行集成测试**:
```bash
# 运行所有集成测试
pytest tests/integration/ -v

# 运行特定集成测试
pytest tests/integration/test_builders.py -v

# 使用标记运行
pytest tests/ -m "integration"
```

### Benchmark测试 (tests/benchmarks/)

#### adapters/ - 模型适配器
| 文件 | 用途 | API? |
|------|------|------|
| mock.py | 测试用Mock适配器 | ✗ |
| zhipu.py | 智谱AI适配器 | ✓ |

#### locomo/ - LOCOMO数据集评估
| 文件 | 用途 | 运行方式 | 需要API? |
|------|------|---------|---------|
| dataset.py | 数据集类定义 | - | ✗ |
| evaluator.py | 评估器实现 | - | ✗ |
| test_e2e_zhipu.py | 端到端测试（支持多种模式） | 见下方说明 | ✓ |

**test_e2e_zhipu.py 运行方式**:
```bash
# 快速测试（1 QA pair）
python tests/benchmarks/locomo/test_e2e_zhipu.py --quick

# 标准测试（5 QA pairs，默认）
python tests/benchmarks/locomo/test_e2e_zhipu.py

# 完整测试（所有 QA pairs）
python tests/benchmarks/locomo/test_e2e_zhipu.py --full

# 调试模式（详细日志，1 QA pair）
python tests/benchmarks/locomo/test_e2e_zhipu.py --debug

# 自定义QA对数量
python tests/benchmarks/locomo/test_e2e_zhipu.py --qa-pairs 10
```

#### verification/ - 功能验证脚本

开发时使用的验证脚本，非pytest测试。

| 文件 | 用途 | 运行方式 | 需要API? |
|------|------|---------|---------|
| compare_embedders.py | 对比不同Embedder性能 | `python tests/benchmarks/verification/compare_embedders.py` | ✓ |
| compare_relation_builders.py | 对比RelationBuilder效果 | `python tests/benchmarks/verification/compare_relation_builders.py` | ✓ |
| test_relation_builder_simple.py | 验证RelationBuilder功能 | `python tests/benchmarks/verification/test_relation_builder_simple.py` | ✓ |

## 测试执行方式

### 方式1: 使用pytest（推荐）

用于运行单元测试和集成测试：

```bash
# 运行所有测试
pytest tests/

# 按目录运行
pytest tests/unit/ -v
pytest tests/integration/ -v

# 运行特定文件
pytest tests/unit/test_core.py -v
pytest tests/integration/test_builders.py -v

# 运行特定测试
pytest tests/integration/test_builders.py::TestRandomRelationBuilder::test_initialization_defaults -v

# 按标记运行
pytest tests/ -m "unit"           # 只运行单元测试
pytest tests/ -m "integration"    # 只运行集成测试
pytest tests/ -m "not api"        # 排除需要API的测试
pytest tests/ -m "not slow"       # 排除慢速测试

# 查看详细输出
pytest tests/ -v -s

# 并行运行（需要pytest-xdist）
pytest tests/ -n auto
```

### 方式2: 直接运行Python脚本

用于运行benchmark测试和验证脚本：

```bash
# LOCOMO benchmark测试（多种模式）
python tests/benchmarks/locomo/test_e2e_zhipu.py --quick
python tests/benchmarks/locomo/test_e2e_zhipu.py --full
python tests/benchmarks/locomo/test_e2e_zhipu.py --debug

# 功能验证脚本
python tests/benchmarks/verification/compare_embedders.py
python tests/benchmarks/verification/compare_relation_builders.py
python tests/benchmarks/verification/test_relation_builder_simple.py
```

## 测试标记

```bash
# 只运行单元测试
pytest tests/ -m "unit"

# 只运行集成测试
pytest tests/ -m "integration"

# 排除需要API的测试
pytest tests/ -m "not api"

# 运行快速测试（排除慢速测试）
pytest tests/ -m "not slow"

# 查看所有标记
pytest --markers
```

**当前标记使用情况**：
- `@pytest.mark.unit`: 单元测试（暂未广泛使用）
- `@pytest.mark.integration`: 集成测试（暂未广泛使用）
- `@pytest.mark.api`: 需要API密钥的测试（暂未广泛使用）
- `@pytest.mark.slow`: 慢速测试（暂未广泛使用）

注意：测试标记需要在conftest.py中配置，当前主要通过运行方式区分测试类型。

## Mock vs 真实API

### 使用Mock

**适用场景**：
- 单元测试
- 集成测试（大部分情况）
- 快速验证逻辑

**优点**：
- 快速
- 不消耗API配额
- 不需要网络连接

**缺点**：
- 无法验证真实API集成
- 无法测试真实embedding质量

**示例**：
```python
from tests.benchmarks.adapters.mock import MockEmbedder

embedder = MockEmbedder()
memory = Memory(embedder=embedder)
```

### 使用真实API

**适用场景**：
- Benchmark测试
- 验证API集成
- 性能评估

**优点**：
- 测试真实场景
- 验证API集成正确性
- 评估实际性能

**缺点**：
- 消耗API配额
- 需要网络连接
- 运行较慢
- 需要API密钥

**示例**：
```python
import os
from tests.benchmarks.adapters.zhipu import ZhipuEmbedder

api_key = os.environ.get("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("ZHIPUAI_API_KEY not set")

embedder = ZhipuEmbedder(api_key=api_key, model="embedding-3")
memory = Memory(embedder=embedder)
```

## 开发新测试

### 测试命名规范

- 单元测试：`test_<module>_<function>.py`
- 集成测试：`test_<feature>.py`
- Benchmark：`benchmarks/<dataset>/test_<model>.py`
- 验证脚本：`verify_<feature>.py` 或 `compare_<features>.py`

### 测试模板

#### pytest测试模板

```python
"""Test <Feature> functionality."""

import pytest
from mempy import <Class to test>

class Test<Feature>:
    """Test suite for <Feature>."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        # Setup code
        pass

    def test_<specific_behavior>(self, setup):
        """Test that <specific behavior> works correctly."""
        # Test code
        assert True
```

#### 验证脚本模板

```python
#!/usr/bin/env python3
"""Verification script for <Feature>.

Usage:
    python tests/verification/verify_<feature>.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def main():
    """Run verification."""
    # Your verification code here
    pass

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

### 使用Mock还是真实API？

**使用Mock**：
- ✅ 单元测试
- ✅ 集成测试（大部分情况）
- ✅ 快速验证逻辑
- ✅ CI/CD自动化测试

**使用真实API**：
- ✅ Benchmark测试
- ✅ 验证API集成
- ✅ 性能评估
- ✅ 发布前验证

## 常见问题

### Q: 测试失败提示缺少API密钥
**A**: 运行以下命令配置API密钥：
```bash
cp tests/benchmarks/.env.example tests/benchmarks/.env
# 编辑 tests/benchmarks/.env 添加你的API密钥
```

或者设置环境变量：
```bash
export ZHIPUAI_API_KEY=your-api-key-here
```

### Q: 如何跳过需要API的测试？
**A**: 使用以下命令：
```bash
# 只运行不需要API的测试
pytest tests/ -m "not api"

# 或者直接运行不使用API的测试文件
pytest tests/test_core.py -v
```

### Q: Benchmark测试很慢，怎么办？
**A**: 使用快速测试选项：
```bash
# 只运行1个QA pair
python tests/benchmarks/locomo/test_e2e_zhipu_debug.py

# 只运行5个QA pairs
python tests/benchmarks/locomo/test_e2e_zhipu.py

# 使用Mock而不是真实API
python tests/benchmarks/locomo/compare_embedders.py  # 会先运行Mock测试
```

### Q: 如何查看详细的测试日志？
**A**:
```bash
# pytest测试
pytest tests/ -v -s

# Python脚本
# 脚本中已经有verbose=True参数，会自动输出详细日志
```

### Q: 测试数据在哪里？
**A**:
- LOCOMO数据集: `tests/benchmarks/data/locomo10.json`
- 其他测试数据: `tests/data/`

### Q: 如何添加新的测试数据？
**A**:
1. 将数据文件放入 `tests/benchmarks/data/` 或 `tests/data/`
2. 确保数据文件不会太大（建议<1MB）
3. 在测试代码中添加数据加载逻辑
4. 更新相关文档

## 测试最佳实践

1. **使用隔离的测试环境**：每个测试应该独立运行，不依赖其他测试
2. **使用fixtures**：在conftest.py中定义共享的测试fixtures
3. **Mock外部依赖**：除了专门测试API集成的测试，其他测试应该使用Mock
4. **清晰的测试命名**：测试名称应该清楚描述测试的内容
5. **添加测试文档**：为复杂的测试添加docstring
6. **保持测试快速**：单元测试应该快速运行，慢速测试标记为`@slow`
7. **使用环境变量管理敏感信息**：永远不要硬编码API密钥

## 下一步

1. **运行测试**：
   ```bash
   pytest tests/ -v
   ```

2. **添加测试标记**：
   在conftest.py中配置测试标记，以便更好地分类测试

3. **统一Mock实现**：
   将所有Mock实现统一到conftest.py，消除重复代码

4. **优化测试结构**：
   考虑将测试分类到unit/和integration/目录

## 相关文档

- [pytest文档](https://docs.pytest.org/)
- [mempy项目README](../../README.md)
- [API密钥配置示例](benchmarks/.env.example)

## 更新日志

- 2025-01-07: 创建测试文档，整理测试目录结构
- 待办: 添加测试标记配置
- 待办: 统一Mock实现
- 待办: 优化目录结构
