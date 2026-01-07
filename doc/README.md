# mempy 开发者文档

本文档面向开发人员和架构师，提供 mempy 的技术细节和架构设计。

**面向对象**: 开发人员、架构师
**用户文档**: 请查看项目根目录的 README.md

---

## 📋 文档导航

### 🏗️ 架构设计

- **[Architecture](architecture.md)** - mempy 核心架构设计
  - 项目定位和设计愿景
  - 核心设计原则
  - 三层架构（应用层/服务层/存储层）
  - 关键设计决策
  - 核心接口设计
  - 实现细节

### 🔌 API 参考

- **[API Reference](api.md)** - 完整的 API 参考
  - Memory 类
  - Embedder 接口
  - MemoryProcessor 接口
  - 数据类和异常

### 🔧 扩展开发

- **[Embedder Adapter Guide](adapter-guide.md)** - 创建自定义嵌入器
  - Embedder 接口规范
  - 适配器实现示例
  - 测试指南

- **[Strategy System](strategy_system.md)** - 策略系统架构设计
  - 三阶段处理流水线
  - MemoryProcessor 接口
  - RelationBuilder 接口
  - 实现细节和最佳实践

- **[RelationBuilder Guide](relation_builder_guide.md)** - 关系构建器使用指南
  - RelationBuilder 接口
  - 内置实现
  - 自定义构建器示例
  - 与 RelationExplorationStrategy 的区别

- **[Custom Strategies Guide](strategies.md)** - 自定义记忆演化策略
  - 策略模式介绍
  - 置信度演化策略
  - 牢固度计算策略
  - 遗忘阈值策略
  - 关系探索策略

### 📊 评估与基准

- **[Benchmark Guide](benchmark.md)** - LOCOMO 基准测试
  - LOCOMO 数据集
  - 评估指标
  - 运行基准测试
  - 结果解读

### ⚡ 快速开始（开发者）

- **[Quick Start](quickstart.md)** - 开发者快速入门
  - 安装和配置
  - 基础用法
  - 示例代码

---

## 🎯 文档使用指南

### 如果你想...

**理解项目架构**
→ 从 [Architecture](architecture.md) 开始

**查找 API 文档**
→ 查看 [API Reference](api.md)

**扩展 mempy 功能**
→ 阅读 [Embedder Adapter Guide](adapter-guide.md)、[Strategy System](strategy_system.md)、[RelationBuilder Guide](relation_builder_guide.md) 或 [Custom Strategies Guide](strategies.md)

**运行基准测试**
→ 参考 [Benchmark Guide](benchmark.md)

**快速上手开发**
→ 查看 [Quick Start](quickstart.md)

---

## 🔗 相关资源

- **GitHub**: [项目仓库](https://github.com/yourusername/mempy)
- **Issues**: [提交问题](https://github.com/yourusername/mempy/issues)
- **Discussions**: [参与讨论](https://github.com/yourusername/mempy/discussions)

---

## 📝 文档维护

本文档持续更新中。如有问题或建议，请提交 Issue 或 PR。
