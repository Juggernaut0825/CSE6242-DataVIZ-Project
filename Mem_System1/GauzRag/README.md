# GauzRag

**基于 GraphRAG 的端到端知识图谱 + 语义召回系统**

## 系统架构

```
原始文档 (Markdown/Text)
    ↓
[阶段 1] Facts 提取 (LLM)
    ↓
MySQL 数据库 (fact_id + content)
    ↓
[阶段 2] 知识图谱构建 (GraphRAG)
    ├─ 实体抽取
    ├─ 关系抽取
    ├─ 社区检测 (Leiden)
    └─ 社区报告生成 (LLM摘要)
    ↓
Knowledge Graph (Parquet 文件)
    ↓
[阶段 3] Community-Facts 映射
    ↓
Community of Facts (JSON)
    ↓
[阶段 4] Embedding 索引构建
    ↓
Community Embeddings (Pickle)
    ↓
[阶段 5] 查询召回
    ↓
相关 Communities + Facts
```

## 核心特点

### 1. 端到端流程

- **自动化**: 从原始文档到可查询的知识库，全自动处理
- **模块化**: 每个阶段可独立运行和调试
- **可追溯**: 从召回结果可追溯到原始 fact 和文档

### 2. 高效召回策略

- **只对 Community Report 做 Embedding**（不是每条 fact）
- 14 个社区 vs 几百条 facts → 效率提升 10x+
- 基于高质量 LLM 摘要的语义理解

### 3. 智能去重

- 语义级别的去重（不是简单的文本匹配）
- 可配置的相似度阈值
- 保留信息多样性

### 4. 灵活配置

- 支持多种 LLM 提供商（OpenRouter, OpenAI, Azure）
- 使用阿里云百炼 text-embedding-v4（中文优化）
- MySQL 持久化存储

## 快速开始

### 1. 安装依赖

```bash
pip install pymysql pandas pyarrow openai scikit-learn numpy requests
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# LLM 配置
GAUZ_LLM_API_BASE=https://openrouter.ai/api/v1
GAUZ_LLM_API_KEY=your_api_key
GAUZ_LLM_MODEL=openai/gpt-4o-mini
GAUZ_LLM_TEMPERATURE=0.3
GAUZ_LLM_MAX_TOKENS=

# Embedding 配置（阿里云百炼）
DASHSCOPE_API_KEY=your_dashscope_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

# MySQL 配置
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=gauzrag
MYSQL_TABLE=facts
```

### 3. 运行完整流程

```python
from pathlib import Path
from GauzRag import GauzRagConfig, GauzRagPipeline

# 加载配置
config = GauzRagConfig.from_env(Path(".env"))

# 创建 Pipeline
pipeline = GauzRagPipeline(config)

# 运行完整流程
pipeline.run_full_pipeline(
    input_file=Path("your_document.md"),
    verbose=False
)

# 查询召回
results = pipeline.search("你的查询问题", top_k=3)
```

### 4. 分步骤运行

```python
# 步骤 1: 提取 Facts
pipeline.extract_and_store_facts(Path("document.md"))

# 步骤 2: 构建知识图谱
pipeline.build_knowledge_graph()

# 步骤 3: 构建 Community-Facts 映射
pipeline.build_community_mapping()

# 步骤 4: 构建 Embedding 索引
pipeline.build_embeddings()

# 步骤 5: 查询
results = pipeline.search("查询问题")
```

## 模块说明

### config.py - 配置管理

- `GauzRagConfig`: 统一的配置类
- 支持从环境变量加载
- 自动创建必要的目录

### database.py - 数据库管理

- `DatabaseManager`: MySQL 数据库操作
- Facts 的 CRUD 操作
- 自动创建表结构

### fact_extractor.py - Facts 提取

- `FactExtractor`: 使用 LLM 从文档提取结构化 facts
- 支持自定义 prompt
- 自动解析提取结果

### graph_builder.py - 图谱构建

- `GraphBuilder`: 基于 GraphRAG 构建知识图谱
- 自动处理 fact_id 元数据
- 支持自定义工作流

### community_mapper.py - Community 映射

- `CommunityMapper`: 将图谱社区映射回 facts
- 逆向追溯：community → text_units → documents → fact_ids
- 生成 JSON 和 Markdown 输出

### embedder.py - Embedding 生成

- `DashScopeEmbedder`: 阿里云百炼 Embedding 封装
- 兼容 SentenceTransformer 接口
- 自动批处理

### searcher.py - 查询召回

- `CommunitySearcher`: 基于余弦相似度的语义召回
- `EmbeddingIndexBuilder`: Embedding 索引构建
- 支持语义去重

### pipeline.py - 主流程

- `GauzRagPipeline`: 端到端流程编排
- 提供高级 API
- 自动管理模块依赖

## 示例代码

查看 `examples/` 目录下的示例：

- `example_full_pipeline.py`: 完整流程示例
- `example_step_by_step.py`: 分步骤执行
- `example_search.py`: 批量查询测试
- `example_query_interactive.py`: 交互式查询工具

## API 文档

### GauzRagPipeline

#### 初始化

```python
pipeline = GauzRagPipeline(config: GauzRagConfig)
```

#### 运行完整流程

```python
pipeline.run_full_pipeline(
    input_file: Path,  # 输入文档路径
    verbose: bool = False  # 是否显示详细日志
)
```

#### 查询召回

```python
results = pipeline.search(
    query: str,  # 查询文本
    top_k: int = 3,  # 返回前 k 个社区
    dedupe: bool = True,  # 是否语义去重
    dedupe_threshold: float = 0.85,  # 去重阈值
    max_facts_per_community: int = 10  # 每个社区最多返回的 facts
)
```

返回结果格式：

```python
[
    {
        "community_id": 2,
        "community_name": "Main_Agent and Its Operational Framework",
        "relevance_score": 0.812,
        "report": {
            "title": "...",
            "summary": "...",
            "rating": 7.5,
            "rating_explanation": "..."
        },
        "fact_count": 10,
        "facts": [
            {"fact_id": 13, "content": "..."},
            {"fact_id": 14, "content": "..."},
            ...
        ]
    },
    ...
]
```

## 性能指标

- **Embedding 索引构建**: < 10 秒（14 个社区）
- **单次查询响应**: < 1 秒
- **Embedding 维度**: 1024（text-embedding-v4）
- **召回精度**: 基于阿里云百炼模型，中文语义理解优秀

## 常见问题

### Q: 为什么使用 MySQL 而不是向量数据库？

A: 因为我们只对 Community Report 做 embedding（数量很少，通常 < 100），不需要向量数据库的 ANN 搜索能力。MySQL 足够轻量且易于管理。当社区数量增长到数百个时，可以考虑迁移到 Milvus/Qdrant。

### Q: 如何添加新的文档？

A: 有两种方式：
1. 追加模式：只提取新文档的 facts，然后重新运行 graph_builder
2. 完全重建：清空数据库，重新提取所有文档的 facts

推荐使用追加模式 + 定期完全重建的策略。

### Q: 如何调整召回质量？

A: 可以调整以下参数：
- `top_k`: 增大可召回更多社区
- `dedupe_threshold`: 降低可保留更多样化的 facts
- `min_score`: 提高可过滤低相关度结果

### Q: 支持哪些 LLM 提供商？

A: 通过 OpenRouter，支持几乎所有主流 LLM（GPT, Claude, Grok, Llama 等）。也可以直接使用 OpenAI 或 Azure OpenAI。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 作者

Gauz - 2025

