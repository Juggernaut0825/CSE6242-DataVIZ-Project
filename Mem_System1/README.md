# Memory System - 智能记忆系统

一个基于知识图谱和向量检索的智能记忆系统，能够理解对话、提取知识、建立关系，并支持智能检索和推理。

---

## 系统特点

### 🧠 双层记忆架构
- **短期记忆**：对话实时存储，立即可查（< 1秒）
- **长期记忆**：后台异步构建为知识图谱 + 向量索引，支持深度语义检索和因果推理

### 📊 知识图谱自动构建
- 自动提取原子事实（Facts）
- **显性关系提取**：识别文本中明确表达的因果、时序、支持等 8 种逻辑关系
- **隐性关系推断**：通过实体共现和 LLM 分析发现潜在联系
- 社区检测自动发现主题聚类

### 🔍 智能检索
- **语义检索**：基于 embedding 的相似度匹配
- **图谱扩展**：沿着因果链、时间线等关系多跳检索
- **时序扩展**：获取对话的前后文上下文
- **意图识别**：使用 LLM 将自然语言转化为具体的结构化查询

---

## 核心接口

### 1. `/extract` - 对话提取与索引

**功能**：将对话内容提取为结构化知识，并建立检索索引。

#### 工作流程

**同步阶段（1-2秒）**：
- 保存对话原文（短期记忆立即可查）
- 返回对话 ID

**异步阶段（后台30-120秒）**：
1. 提取原子事实（Facts）
2. **提取显性关系** - 识别事实间的明确逻辑关系（NEW ✨）
3. 构建知识图谱（实体 + 隐性关系）
4. 生成主题聚类
5. 构建向量索引
6. 标记为长期记忆

#### 请求示例

```json
POST /extract
{
  "text": "昨天下雨导致路面湿滑，因此发生了交通事故。交警及时赶到现场疏导交通。",
  "project_id": "my_project",
  "metadata": {
    "username": "alice",
    "timestamp": "2024-12-25T10:00:00Z"
  }
}
```

#### 系统会自动识别

**提取的 Facts**：
- Fact 1: "昨天下雨"
- Fact 2: "路面湿滑"
- Fact 3: "发生了交通事故"
- Fact 4: "交警赶到现场"

**显性关系（自动识别）**：
- Fact 1 --**Cause**--> Fact 2 （下雨导致湿滑）
- Fact 2 --**Cause**--> Fact 3 （湿滑导致事故）
- Fact 3 --**Temporal**--> Fact 4 （事故之后交警赶到）

**响应**：
```json
{
  "message": "对话已接收（短期记忆），后台正在构建长期记忆",
  "conversation_id": 42,
  "project_id": "my_project"
}
```

#### 替换模式（文件块更新）

当需要更新文件内容时，使用 `replace: true` 模式进行硬裁剪替换：

```json
POST /extract
{
  "text": "更新后的文档内容...",
  "project_id": "my_project",
  "content_type": "file_chunk",
  "replace": true,
  "metadata": {
    "file_hash": "sha256:abc123...",
    "chunk_index": 0,
    "file_path": "docs/readme.md"
  }
}
```

**工作原理**：
1. 通过 `file_hash + chunk_index` 定位旧数据
2. 删除旧对话、Facts、图谱关系、向量索引
3. 插入新内容（完整索引流程）

**适用场景**：
- 文档内容更新（Markdown、代码文件等）
- 知识库同步
- 版本化内容管理

---

### 2. `/search` - 三级并行检索

**功能**：智能检索系统的核心接口，支持多维度检索和关系扩展。

#### 三种检索模式

**模式 1：基础语义检索**
只需提供查询文本，系统自动进行语义匹配：
```json
{
  "query": "用户压力大",
  "project_id": "my_project"
}
```

**模式 2：硬过滤检索**
使用时间范围或元数据进行精确过滤：
```json
{
  "query": "工作相关的对话",
  "project_id": "my_project",
  "filters": {
    "time_range": {
      "start": "2024-12-01T00:00:00Z",
      "end": "2024-12-25T23:59:59Z"
    },
    "metadata": {
      "username": "alice"
    }
  }
}
```

**模式 3：扩展检索（核心功能）**
通过图谱关系或时序关系进行多跳扩展：

```json
POST /search
{
  "query": "为什么用户最近压力很大？",
  "project_id": "my_project",
  "top_k": 5,
  "use_refine": true,
  
  "expansions": {
    "graph": {
      "enabled": true,
      "max_hops": 2,
      "relation_types": ["CAUSE", "SUPPORT"]
    },
    "temporal": {
      "enabled": true,
      "mode": "turn",
      "hop_distance": 1,
      "direction": "backward"
    }
  }
}
```

#### 关系类型说明

- **CAUSE** - 因果关系："为什么"、"原因"场景
- **TEMPORAL** - 时间顺序："之后"、"接着"场景
- **SUPPORT** - 支持证据："证据"、"支持"场景
- **ELABORATE** - 详细阐述："详细"、"具体"场景
- **CONTRADICT** - 矛盾冲突："矛盾"、"相反"场景
- **CONDITION** - 条件关系："如果"、"假设"场景
- **PURPOSE** - 目的关系："目的"、"为了"场景
- **ANALOGY** - 类比关系："类似"、"像"场景

#### 响应示例（use_refine=true）

```json
{
  "query": "为什么用户最近压力很大？",
  "project_id": "my_project",
  "bundles": [
    {
      "bundle_id": 1,
      "related_memory": "用户近期压力源于同时承接三个项目，且截止日期集中在本周，导致时间紧张。",
      "quote": "我现在有三个项目要同时推进",
      "conversations": [...],
      "facts": [
        {
          "fact_id": 101,
          "content": "用户接了三个项目",
          "relevance_score": 0.92,
          "hop_facts": {
            "1": [
              {
                "fact_id": 102,
                "content": "项目截止日期是本周五",
                "relation": "TEMPORAL"
              }
            ]
          }
        }
      ],
      "topics": [...]
    }
  ],
  "total_bundles": 1,
  "refined": true
}
```

**字段说明**：
- `related_memory`: LLM 整合后的高密度摘要
- `quote`: 引用的关键原文
- `conversations/facts/topics`: 保留原始数据用于溯源
- `hop_facts`: 图谱扩展的多跳关联事实（按跳数分层）

#### Bundle 聚合机制

系统会自动将相关的记忆片段聚合为 **Bundle**，每个 Bundle 包含：
- **Facts**：核心事实 + 通过关系扩展的相关事实
- **Conversations**：原始对话上下文
- **Topics**：主题摘要（如果有）

---

### 3. `/agenticSearch` - 智能搜索

**功能**：用自然语言描述需求，系统自动理解意图并构建最优查询策略。

#### 适用场景

- **时间推理**："上周提到的那件事后来怎么样了？"
- **因果分析**："为什么会发生这种情况？"
- **上下文查询**："用户说完那句话后又说了什么？"
- **主题检索**："关于家庭的所有对话"

#### 请求示例

```json
POST /agenticSearch
{
  "query": "用户上周提到孩子生病后说了什么？",
  "project_id": "my_project",
  "use_refine": true
}
```

#### 系统自动处理

**意图识别**：
- 识别查询类型：时序上下文查询
- 提取时间范围：上周（自动计算为具体日期）
- 提取关键词：孩子、生病、儿童、健康
- 确定扩展策略：时序扩展（向前查找后续对话）

**自动构建结构化查询** → 执行搜索 → 返回结果

#### 响应示例

```json
{
  "original_query": "用户上周提到孩子生病后说了什么？",
  "interpreted_intent": "查找用户提到孩子生病后的后续对话内容",
  "structured_query": {
    "query": "孩子 生病 儿童 健康",
    "project_id": "my_project",
    "top_k": 5,
    "filters": {
      "time_range": {
        "start": "2024-12-18T00:00:00Z",
        "end": "2024-12-25T23:59:59Z"
      }
    },
    "expansions": {
      "temporal": {
        "enabled": true,
        "mode": "turn",
        "hop_distance": 2,
        "direction": "forward"
      }
    }
  },
  "search_results": {
    "query": "孩子 生病 儿童 健康",
    "project_id": "my_project",
    "bundles": [
      {
        "bundle_id": 1,
        "facts": [
          {
            "fact_id": 201,
            "content": "孩子发烧38度",
            "relevance_score": 0.95
          },
          {
            "fact_id": 202,
            "content": "已经带孩子去医院",
            "relevance_score": 0.88
          },
          {
            "fact_id": 203,
            "content": "医生说是普通感冒",
            "relevance_score": 0.82
          }
        ]
      }
    ],
    "total_bundles": 1
  }
}
```

---

## 核心概念

### 短期记忆 vs 长期记忆

| 特性 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| 可用时间 | 立即（< 1秒） | 30-120秒后 |
| 查询方式 | 关键词匹配 | 语义检索 + 图推理 |
| 适用场景 | 最近对话回顾 | 深度分析、因果推理 |

### 显性关系 vs 隐性关系

**显性关系**：文本中明确表达的逻辑关系
- 例如："因为下雨，所以路滑"（CAUSE）
- 提取时机：Facts 提取后立即识别
- 置信度：通常 > 0.8

**隐性关系**：通过实体共现推断的潜在联系
- 例如：两个 Facts 都提到"GPT-4"（实体共现）
- 提取时机：图谱构建阶段
- 置信度：通常 0.6-0.8

---

## 快速开始

### 1. 提取对话

```bash
curl -X POST http://localhost:1234/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天讨论了新项目预算，客户希望降低成本，但我们需要保证质量。这是个挑战。",
    "project_id": "demo"
  }'
```

### 2. 等待索引（约 30-60 秒）

```bash
curl http://localhost:1234/conversation/1/status
```

### 3. 智能搜索

```bash
curl -X POST http://localhost:1234/agenticSearch \
  -H "Content-Type: application/json" \
  -d '{
    "query": "为什么降低成本是个挑战？",
    "project_id": "demo",
    "use_refine": true
  }'
```

---

## 技术栈

- **向量数据库**：Qdrant
- **图数据库**：Neo4j
- **关系数据库**：MySQL
- **Embedding**：DashScope text-embedding-v3
- **LLM**：支持 OpenAI API 兼容接口

---

## 性能特点

- **实时响应**：短期记忆 < 1秒
- **异步索引**：长期记忆 30-120秒
- **高效检索**：向量检索 < 100ms，图检索 < 500ms

---

## 示例与文档

### 示例代码
- `examples/example_explicit_relations.py` - 显性关系提取演示
- `examples/example_api_client.py` - API 基础调用
- `examples/example_fact_search.py` - 检索示例

### 详细文档
- [显性关系提取功能](docs/Explicit_Relations_Feature.md)
- [三级检索 API](docs/THREE_LEVEL_SEARCH_API.md)
- [时间维度检索](docs/TIME_DIMENSION_GUIDE.md)

---

## License

MIT

