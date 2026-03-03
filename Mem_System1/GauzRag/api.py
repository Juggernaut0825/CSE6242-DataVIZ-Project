"""
GauzRag FastAPI 服务
支持多项目隔离和查询召回
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np

from .config import GauzRagConfig
from .pipeline import GauzRagPipeline


# ===== 请求/响应模型 =====

# ===== 新规范：嵌套模型定义 =====

class TimeRangeFilter(BaseModel):
    """时间范围过滤"""
    start: Optional[str] = Field(None, description="开始时间（ISO 8601），如 '2024-01-01T00:00:00'")
    end: Optional[str] = Field(None, description="结束时间（ISO 8601），如 '2024-12-31T23:59:59'")


class SearchFilters(BaseModel):
    """硬过滤条件（基于 metadata 的精确过滤）"""
    time_range: Optional[TimeRangeFilter] = Field(None, description="时间范围过滤")
    metadata: Optional[dict] = Field(None, description="元数据过滤条件，如 {'username': 'user_123', 'turn': 5}")


class GraphExpansionConfig(BaseModel):
    """图谱扩展配置（基于 Neo4j 语义关系）"""
    enabled: bool = Field(False, description="是否启用图谱扩展")
    max_hops: int = Field(2, description="最大跳数（1=单跳，2=双跳，默认2）")
    relation_types: Optional[List[str]] = Field(
        None, 
        description="指定关系类型（如 ['CAUSE', 'TEMPORAL']），为空则使用全部 8 种关系"
    )


class TemporalExpansionConfig(BaseModel):
    """时序扩展配置（基于时间/轮次邻近）"""
    enabled: bool = Field(False, description="是否启用时序扩展")
    mode: str = Field("turn", description="扩展模式：'turn'=轮次，'time'=时间窗口")
    hop_distance: int = Field(1, description="扩展跳数：1=±1轮/5分钟，2=±2轮/10分钟")
    direction: str = Field("both", description="扩展方向：'both'=双向，'forward'=向后，'backward'=向前")


class SearchExpansions(BaseModel):
    """扩展检索条件（基于关系的上下文扩展）"""
    graph: Optional[GraphExpansionConfig] = Field(None, description="图谱扩展配置")
    temporal: Optional[TemporalExpansionConfig] = Field(None, description="时序扩展配置")


class QueryRequest(BaseModel):
    """
    查询请求（规范化版本）
    
    三种检索模式：
    1. 基础语义检索：只提供 query + project_id
    2. 硬过滤检索：使用 filters 字段（时间/元数据过滤）
    3. 扩展检索：使用 expansions 字段（图谱/时序扩展）
    """
    # 基础参数
    query: str = Field(..., description="用户查询文本")
    project_id: str = Field(..., description="项目 ID")
    top_k: Optional[int] = Field(3, description="返回前 k 个相关结果")
    use_refine: Optional[bool] = Field(False, description="是否使用 LLM 精炼 Bundle 内容")
    
    # 混合检索参数
    use_bm25: Optional[bool] = Field(False, description="是否启用BM25检索（混合检索模式）")
    bm25_weight: Optional[float] = Field(0.3, description="BM25权重（0-1），向量权重=1-bm25_weight")
    fusion_method: Optional[str] = Field("rrf", description="融合方法：'rrf'（RRF算法）或'weighted'（加权融合）")
    
    # 硬过滤条件（类型2）
    filters: Optional[SearchFilters] = Field(None, description="硬过滤条件（时间/元数据）")
    
    # 扩展检索条件（类型3）
    expansions: Optional[SearchExpansions] = Field(None, description="扩展检索条件（图谱/时序）")
    
    # ===== 向后兼容：保留旧字段（已废弃，优先使用 filters 和 expansions）=====
    use_graph_expansion: Optional[bool] = Field(None, description="[已废弃] 使用 expansions.graph.enabled 代替")
    max_hops: Optional[int] = Field(None, description="[已废弃] 使用 expansions.graph.max_hops 代替")
    metadata_filter: Optional[dict] = Field(None, description="[已废弃] 使用 filters.metadata 代替")
    start_time: Optional[str] = Field(None, description="[已废弃] 使用 filters.time_range.start 代替")
    end_time: Optional[str] = Field(None, description="[已废弃] 使用 filters.time_range.end 代替")
    use_temporal_expansion: Optional[bool] = Field(None, description="[已废弃] 使用 expansions.temporal.enabled 代替")
    temporal_mode: Optional[str] = Field(None, description="[已废弃] 使用 expansions.temporal.mode 代替")
    temporal_hop_distance: Optional[int] = Field(None, description="[已废弃] 使用 expansions.temporal.hop_distance 代替")
    temporal_direction: Optional[str] = Field(None, description="[已废弃] 使用 expansions.temporal.direction 代替")


class AgenticSearchRequest(BaseModel):
    """
    Agentic 搜索请求（自然语言转结构化查询）
    
    使用 LLM 将自然语言查询转化为结构化的 /search 参数
    """
    query: str = Field(..., description="自然语言查询（如：'帮我找一下用户A在12月关于工作的对话'）")
    project_id: str = Field(..., description="项目 ID")
    top_k: Optional[int] = Field(5, description="返回前 k 个相关结果（默认 5）")
    
    # 可选：用户提供的上下文信息
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息（如当前日期、用户名等）")


class FactQueryRequest(BaseModel):
    """Fact 查询请求"""
    query: str = Field(..., description="用户查询文本")
    metadata_filter: Optional[dict] = Field(None, description="元数据过滤条件")
    project_id: str = Field(..., description="项目 ID")
    top_k: Optional[int] = Field(10, description="返回前 k 个相关 facts")
    include_community: Optional[bool] = Field(True, description="是否包含所属社区的 report")
    refine: Optional[bool] = Field(True, description="是否使用 LLM 将召回的 facts 整合成自然语言回答（默认开启）")
    start_time: Optional[str] = Field(None, description="开始时间（ISO 格式），如 '2024-01-01T00:00:00'")
    end_time: Optional[str] = Field(None, description="结束时间（ISO 格式），如 '2024-12-31T23:59:59'")


class TimeDimensionQueryRequest(BaseModel):
    """时间维度查询请求"""
    query: str = Field(..., description="用户查询文本")
    project_id: str = Field(..., description="项目 ID")
    start_time: str = Field(..., description="开始时间（ISO 格式），如 '2024-01-01T00:00:00'")
    end_time: str = Field(..., description="结束时间（ISO 格式），如 '2024-12-31T23:59:59'")
    top_k: Optional[int] = Field(10, description="返回前 k 个相关 facts")
    use_graph: Optional[bool] = Field(True, description="是否使用 Neo4j 图检索")
    use_vector: Optional[bool] = Field(True, description="是否使用 Qdrant 向量检索")
    entity_filter: Optional[str] = Field(None, description="实体过滤（仅用于图检索）")
    metadata_filter: Optional[dict] = Field(None, description="元数据过滤（仅用于向量检索）")


class TimeDimensionQueryResponse(BaseModel):
    """时间维度查询响应"""
    query: str
    project_id: str
    start_time: str
    end_time: str
    graph_results_count: int
    vector_results_count: int
    merged_results: List[Dict[str, Any]]
    total_results: int
    time_distribution: Optional[List[Dict[str, Any]]] = None


class ExtractRequest(BaseModel):
    """Facts 提取请求"""
    text: str = Field(..., description="输入文本（普通文本或图片描述）")
    project_id: str = Field(..., description="项目 ID")
    source_name: Optional[str] = Field(None, description="源文件名（向后兼容）")
    image_url: Optional[str] = Field(None, description="图片URL（如果提供，则text为图片描述，将直接作为fact存储，不进行LLM提取）")
    content_type: Optional[str] = Field("conversation", description="内容类型: 'conversation' 或 'file_chunk'")
    replace: Optional[bool] = Field(False, description="是否替换模式：通过 metadata.file_hash + chunk_index 查找并删除旧数据")
    metadata: Optional[dict] = Field(None, description="元数据（根据content_type包含不同字段）")


class FactResponse(BaseModel):
    """Fact 响应"""
    fact_id: int
    content: str


class TopicReport(BaseModel):
    """Topic 报告"""
    topic_id: int
    title: str
    summary: str


class ConversationItem(BaseModel):
    """对话项（在Bundle中）"""
    conversation_id: int
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class FactItem(BaseModel):
    """Fact项（在Bundle中）"""
    fact_id: int
    content: str
    score: float
    image_url: Optional[str] = None
    hop_facts: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="多跳扩展的Facts（仅种子节点有此字段）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class TopicItem(BaseModel):
    """Topic项（在Bundle中）"""
    topic_id: int
    title: str
    summary: str
    score: float


class BundleResponse(BaseModel):
    """Bundle响应（关系网络聚合结果）"""
    bundle_id: int
    conversations: List[ConversationItem] = Field(default_factory=list, description="Bundle中的对话")
    facts: List[FactItem] = Field(default_factory=list, description="Bundle中的Facts")
    topics: List[TopicItem] = Field(default_factory=list, description="Bundle中的Topics")


class RefinedBundleResponse(BaseModel):
    """精炼后的Bundle响应"""
    bundle_id: int
    related_memory: str = Field(description="LLM提炼的与查询相关的记忆")
    quote: Optional[str] = Field(default=None, description="引用的关键原文")
    # 保留原始数据
    conversations: List[ConversationItem] = Field(default_factory=list, description="Bundle中的对话")
    facts: List[FactItem] = Field(default_factory=list, description="Bundle中的Facts")
    topics: List[TopicItem] = Field(default_factory=list, description="Bundle中的Topics")


class FactResultResponse(BaseModel):
    """Fact检索结果（三级检索：Fact + Conversation + Topics）- 保留用于其他接口"""
    fact_id: int
    fact_content: str
    relevance_score: float
    conversation_text: str  # 该Fact所属的对话原文
    topics: List[TopicReport] = Field(default_factory=list, description="该Fact所属的Topics报告")


class CommunityResponse(BaseModel):
    """Community 响应（保留用于其他接口）"""
    community_id: int
    community_name: str
    relevance_score: float
    report: str | Dict[str, Any]
    fact_count: int
    facts: List[FactResponse]
    topics: Optional[List[TopicReport]] = Field(default_factory=list, description="Facts所属的Topics报告")


class ShortTermMemory(BaseModel):
    """短期记忆响应"""
    related_memory: Optional[str] = Field(default=None, description="LLM提炼的短期记忆摘要（如果use_refine=True）")
    conversations: List[Dict[str, Any]] = Field(default_factory=list, description="未索引的对话")
    total_conversations: int = 0


class RecentTurnsBundle(BaseModel):
    """最近对话轮次Bundle（多轮对话上下文）"""
    conversations: List[ConversationItem] = Field(default_factory=list, description="最近N轮的对话")
    related_memory: Optional[str] = Field(default=None, description="LLM提炼的对话上下文总结")


class GraphExpansionInfo(BaseModel):
    """图谱扩展信息（简化版统计，详细信息在 Facts 的 hop_facts 字段中）"""
    enabled: bool = Field(..., description="是否启用图谱扩展")
    seed_facts_count: int = Field(..., description="种子 Facts 数量")
    total_expanded_facts: int = Field(..., description="总扩展 Facts 数量")
    max_hops_configured: int = Field(..., description="配置的最大跳数")
    actual_hops_reached: int = Field(..., description="实际达到的最大跳数")
    by_hop_level: Dict[str, int] = Field(..., description="按跳数统计")


class BundleQueryResponse(BaseModel):
    """Bundle查询响应"""
    query: str
    project_id: str
    short_term_memory: Optional[ShortTermMemory] = Field(default=None, description="短期记忆（未索引的数据）")
    recent_turns: Optional[RecentTurnsBundle] = Field(default=None, description="最近N轮对话（多轮对话上下文）")
    bundles: List[BundleResponse] | List[RefinedBundleResponse]
    total_bundles: int
    refined: bool = Field(default=False, description="是否经过LLM精炼")
    graph_expansion: Optional[GraphExpansionInfo] = Field(default=None, description="图谱扩展信息（多跳推理路径）")
    temporal_expansion: Optional[Dict[str, Any]] = Field(default=None, description="时序扩展信息（时间/轮次维度）")


class AgenticSearchResponse(BaseModel):
    """Agentic 搜索响应"""
    original_query: str = Field(..., description="原始自然语言查询")
    interpreted_intent: str = Field(..., description="LLM 理解的查询意图")
    structured_query: QueryRequest = Field(..., description="转化后的结构化查询")
    search_results: BundleQueryResponse = Field(..., description="实际搜索结果")


class QueryResponse(BaseModel):
    """查询响应（保留用于其他接口）"""
    query: str
    project_id: str
    results: List[FactResultResponse]
    total_results: int


class FactSearchResult(BaseModel):
    """Fact 搜索结果"""
    fact_id: int
    content: str
    relevance_score: float
    community_id: Optional[int] = None
    community_name: Optional[str] = None
    community_report: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据（包含来源信息）")


class FactQueryResponse(BaseModel):
    """Fact 查询响应"""
    query: str
    project_id: str
    results: Optional[List[FactSearchResult]] = Field(None, description="原始 Facts 列表（当 refine=False 时返回）")
    total_facts: int
    refined_answer: Optional[str] = Field(None, description="LLM 整合后的自然语言回答（当 refine=True 时返回）")


class ExtractResponse(BaseModel):
    """Facts 提取响应"""
    project_id: str
    facts_count: int
    message: str


class ProjectStatusResponse(BaseModel):
    """项目状态响应"""
    project_id: str
    facts_count: int
    has_embeddings: bool
    has_community_mapping: bool
    ready_for_search: bool


class FactRelationResponse(BaseModel):
    """Fact 关系响应"""
    fact_id: int
    content: str
    outgoing_relations: List[Dict[str, Any]]
    incoming_relations: List[Dict[str, Any]]
    total_relations: int


# ===== FastAPI 应用 =====

class GauzRagAPI:
    """GauzRag API 服务"""
    
    def __init__(self, base_config: GauzRagConfig):
        """
        初始化 API 服务
        
        Args:
            base_config: 基础配置
        """
        self.base_config = base_config
        self.pipelines: Dict[str, GauzRagPipeline] = {}
        
        # 创建线程池用于后台任务（真正的并发执行）
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg_task")
        
        # 创建 FastAPI 应用
        self.app = FastAPI(
            title="GauzRag API",
            description="基于 GauzRag 的知识图谱 + 语义召回服务",
            version="1.0.0"
        )
        
        # 添加 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境应该限制具体域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self._register_routes()
        
        # 注册应用关闭事件
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """应用关闭时清理资源"""
            print("\n[SHUTDOWN] 正在关闭应用，清理资源...")
            for project_id, pipeline in self.pipelines.items():
                if pipeline.config.graph_engine == "lightrag":
                    try:
                        lightrag_builder = pipeline.get_graph_builder()
                        if lightrag_builder and lightrag_builder.initialized:
                            await lightrag_builder.aclose()
                            print(f"  ✓ 已关闭 {project_id} 的 GauzRag 资源")
                    except Exception as e:
                        print(f"  ⚠️  关闭 {project_id} 的 GauzRag 资源失败: {e}")
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            print("  ✓ 已关闭线程池")
            print("[SHUTDOWN] 资源清理完成\n")
    
    def _get_pipeline(self, project_id: str) -> GauzRagPipeline:
        """
        获取或创建指定项目的 Pipeline
        
        Args:
            project_id: 项目 ID
        
        Returns:
            GauzRagPipeline 实例
        """
        if project_id not in self.pipelines:
            # 为该项目创建独立的配置
            config = GauzRagConfig(
                project_root=self.base_config.project_root,
                llm_provider=self.base_config.llm_provider,
                llm_api_base=self.base_config.llm_api_base,
                llm_api_key=self.base_config.llm_api_key,
                llm_model=self.base_config.llm_model,
                llm_temperature=self.base_config.llm_temperature,
                llm_max_tokens=self.base_config.llm_max_tokens,
                embedding_api_key=self.base_config.embedding_api_key,
                embedding_base_url=self.base_config.embedding_base_url,
                embedding_model=self.base_config.embedding_model,
                mysql_host=self.base_config.mysql_host,
                mysql_port=self.base_config.mysql_port,
                mysql_user=self.base_config.mysql_user,
                mysql_password=self.base_config.mysql_password,
                mysql_database=self.base_config.mysql_database,
                mysql_table=self.base_config.mysql_table,  # 统一使用同一个表
                qdrant_mode=self.base_config.qdrant_mode,
                qdrant_url=self.base_config.qdrant_url,
                # 图谱引擎配置（从 base_config 继承）
                graph_engine=self.base_config.graph_engine,
                neo4j_uri=self.base_config.neo4j_uri,
                neo4j_user=self.base_config.neo4j_user,
                neo4j_password=self.base_config.neo4j_password,
                lightrag_graph_storage=self.base_config.lightrag_graph_storage,
                lightrag_vector_storage=self.base_config.lightrag_vector_storage,
                lightrag_kv_storage=self.base_config.lightrag_kv_storage,
                output_dir=self.base_config.project_root / "output" / project_id,
                cache_dir=self.base_config.project_root / "cache" / project_id,
                logs_dir=self.base_config.project_root / "logs" / project_id,
            )
            
            config.setup_directories()
            # 传入 project_id 用于数据隔离
            self.pipelines[project_id] = GauzRagPipeline(config, project_id=project_id)
        
        return self.pipelines[project_id]
    
    def _refine_facts_with_llm(self, query: str, facts: List[FactSearchResult]) -> str:
        """
        使用 LLM 将召回的 facts 整合成自然语言回答
        
        Args:
            query: 用户查询
            facts: 召回的 facts 列表
        
        Returns:
            LLM 生成的自然语言回答
        """
        from openai import OpenAI
        
        # 构建 facts 上下文
        facts_context = ""
        for i, fact in enumerate(facts, 1):
            facts_context += f"{i}. {fact.content}\n"
            if fact.community_report:
                summary = fact.community_report.get('summary', '')
                # 跳过 Uncategorized 的通用 summary
                if summary and "这个特殊社区包含了" not in summary:
                    facts_context += f"   所属主题: {summary[:100]}...\n"
        
        # 构建 Prompt（严格约束，防止编造）
        prompt = f"""你是一个严格的知识助手。你的任务是整合检索到的事实（Facts）来回答用户问题。

【重要约束】
⚠️ 你只能使用下面【检索到的事实】中的信息
⚠️ 严禁编造、推测、或添加任何事实之外的内容
⚠️ 如果事实不足以完整回答问题，明确说明"基于现有信息..."
⚠️ 不要使用你的训练知识，只使用提供的事实

【用户问题】
{query}

【检索到的事实】
{facts_context}

【任务】
1. 仔细阅读上述所有事实
2. 只使用这些事实中的信息来回答问题
3. 用自然、连贯的语言整合这些事实
4. 如果多个事实相关，合理组织它们的逻辑关系
5. 保持客观，不添加主观判断
6. 如果事实不完全匹配问题，在回答开头说明："根据检索到的相关信息..."

【严禁】
❌ 不要添加事实中没有的信息
❌ 不要做出推测或猜测
❌ 不要使用你的背景知识
❌ 不要编造细节或数据

【你的回答】"""

        try:
            # 调用 LLM
            client = OpenAI(
                api_key=self.base_config.llm_api_key,
                base_url=self.base_config.llm_api_base
            )
            
            response = client.chat.completions.create(
                model=self.base_config.llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个严格的事实整合助手。你只能基于提供的事实回答问题，严禁编造或推测任何信息。如果事实不足以回答，你必须明确说明。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # 降低温度，减少创造性，提高准确性
                max_tokens=self.base_config.llm_max_tokens or 1000
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # 如果 LLM 调用失败，返回一个后备答案
            return f"基于检索到的 {len(facts)} 条相关信息，但 LLM 生成回答时出错: {str(e)}"
    
    async def _delete_conversation_cascade(
        self,
        conversation_id: int,
        project_id: str,
        pipeline
    ) -> Dict[str, Any]:
        """
        级联删除一个 conversation 及其所有关联数据（用于替换模式）
        
        删除顺序：
        1. Qdrant facts
        2. Qdrant conversation
        3. Neo4j Fact 节点
        4. 数据库 facts
        5. 数据库 conversation
        
        Args:
            conversation_id: 对话 ID
            project_id: 项目 ID
            pipeline: GauzRagPipeline 实例
            
        Returns:
            删除结果统计
        """
        from GauzRag.vector_store import GauzRagVectorStore
        from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchAny
        
        print(f"\n[删除旧数据] conversation_id={conversation_id}")
        
        db = pipeline.setup_database()
        
        # 步骤1：查询关联的 fact_ids
        conv_data = db.get_conversation_with_facts(conversation_id)
        fact_ids = [f['fact_id'] for f in conv_data.get('facts', [])] if conv_data else []
        
        print(f"  - 找到 {len(fact_ids)} 个关联的 facts")
        
        deleted_facts_qdrant = 0
        deleted_conv_qdrant = 0
        deleted_facts_neo4j = 0
        deleted_facts_db = 0
        deleted_conv_db = 0
        
        # 步骤2：删除 Qdrant facts
        if fact_ids:
            try:
                with GauzRagVectorStore(
                    persist_directory=pipeline.config.output_dir / "qdrant_db",
                    project_id=project_id,
                    use_server=(pipeline.config.qdrant_mode == "server"),
                    server_url=pipeline.config.qdrant_url
                ) as vector_store:
                    # 删除 facts（通过 fact_id）
                    vector_store.client.delete(
                        collection_name=f"{project_id}_facts",
                        points_selector=FilterSelector(
                            filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="fact_id",
                                        match=MatchAny(any=fact_ids)
                                    )
                                ]
                            )
                        )
                    )
                    deleted_facts_qdrant = len(fact_ids)
                    print(f"  ✓ 已删除 Qdrant facts: {deleted_facts_qdrant} 个")
            except Exception as e:
                print(f"  ⚠️  删除 Qdrant facts 失败: {e}")
        
        # 步骤3：删除 Qdrant conversation
        try:
            with GauzRagVectorStore(
                persist_directory=pipeline.config.output_dir / "qdrant_db",
                project_id=project_id,
                use_server=(pipeline.config.qdrant_mode == "server"),
                server_url=pipeline.config.qdrant_url
            ) as vector_store:
                vector_store.client.delete(
                    collection_name=f"{project_id}_conversations",
                    points_selector=[conversation_id]  # 直接通过 ID 删除
                )
                deleted_conv_qdrant = 1
                print(f"  ✓ 已删除 Qdrant conversation: {conversation_id}")
        except Exception as e:
            print(f"  ⚠️  删除 Qdrant conversation 失败: {e}")
        
        # 步骤4：删除 Neo4j Fact 节点（通过 conversation_id）
        if fact_ids:
            try:
                # 使用 GauzRagGraphBuilder 的 Neo4j store
                from GauzRag.lightrag_graph_builder import GauzRagGraphBuilder
                from pathlib import Path
                
                lightrag_builder = GauzRagGraphBuilder(
                    working_dir=Path(pipeline.config.output_dir),
                    neo4j_uri=pipeline.config.neo4j_uri,
                    neo4j_user=pipeline.config.neo4j_user,
                    neo4j_password=pipeline.config.neo4j_password,
                    project_id=project_id,
                    llm_api_key=pipeline.config.llm_api_key,
                    llm_api_base=pipeline.config.llm_api_base,
                    llm_model=pipeline.config.llm_model
                )
                
                # 初始化（创建 neo4j_store 实例）
                await lightrag_builder.initialize()
                neo4j_store = lightrag_builder.neo4j_store
                
                # 先查询是否存在这些节点
                check_query = """
                MATCH (f:Fact)
                WHERE f.conversation_id = $conversation_id
                RETURN count(f) as count, collect(f.fact_id)[0..5] as sample_fact_ids
                """
                print(f"  [Neo4j] 检查是否存在 Fact 节点: conversation_id={conversation_id}")
                
                async with neo4j_store.driver.session() as session:
                    check_result = await session.run(check_query, {
                        "conversation_id": conversation_id
                    })
                    check_records = [record async for record in check_result]
                
                if check_records:
                    existing_count = check_records[0]['count']
                    sample_ids = check_records[0].get('sample_fact_ids', [])
                    print(f"  [Neo4j] 找到 {existing_count} 个 Fact 节点")
                    if sample_ids:
                        print(f"  [Neo4j] 示例 fact_ids: {sample_ids}")
                else:
                    print(f"  [Neo4j] 查询结果为空")
                
                # 执行删除
                delete_query = """
                MATCH (f:Fact)
                WHERE f.conversation_id = $conversation_id
                DETACH DELETE f
                RETURN count(f) as deleted_count
                """
                print(f"  [Neo4j] 执行删除...")
                
                async with neo4j_store.driver.session() as session:
                    delete_result = await session.run(delete_query, {
                        "conversation_id": conversation_id
                    })
                    delete_records = [record async for record in delete_result]
                
                print(f"  [Neo4j] 删除结果: {delete_records}")
                deleted_facts_neo4j = delete_records[0]['deleted_count'] if delete_records else 0
                print(f"  ✓ 已删除 Neo4j Fact 节点: {deleted_facts_neo4j} 个")
                
                # 关闭连接
                await neo4j_store.close()
                
            except Exception as e:
                import traceback
                print(f"  ⚠️  删除 Neo4j 节点失败: {e}")
                print(f"  详细错误:\n{traceback.format_exc()}")
        else:
            print(f"  ⓘ 跳过 Neo4j 删除（没有 fact_ids）")
        
        # 步骤5：删除数据库 facts
        if fact_ids:
            try:
                conn = db.get_connection()
                try:
                    with conn.cursor() as cur:
                        placeholders = ','.join(['%s'] * len(fact_ids))
                        sql = f"DELETE FROM {db.table} WHERE fact_id IN ({placeholders})"
                        cur.execute(sql, fact_ids)
                        conn.commit()
                        deleted_facts_db = len(fact_ids)
                        print(f"  ✓ 已删除数据库 facts: {deleted_facts_db} 行")
                finally:
                    conn.close()
            except Exception as e:
                print(f"  ⚠️  删除数据库 facts 失败: {e}")
        
        # 步骤6：删除数据库 conversation
        try:
            conn = db.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM conversations WHERE conversation_id = %s", (conversation_id,))
                    conn.commit()
                    deleted_conv_db = 1
                    print(f"  ✓ 已删除数据库 conversation: {conversation_id}")
            finally:
                conn.close()
        except Exception as e:
            print(f"  ⚠️  删除数据库 conversation 失败: {e}")
        
        print(f"[删除完成] conversation_id={conversation_id}\n")
        
        return {
            "deleted_conversation_id": conversation_id,
            "deleted_facts_count": len(fact_ids),
            "details": {
                "qdrant_facts": deleted_facts_qdrant,
                "qdrant_conversation": deleted_conv_qdrant,
                "neo4j_facts": deleted_facts_neo4j,
                "db_facts": deleted_facts_db,
                "db_conversation": deleted_conv_db
            }
        }
    
    def _save_indexing_timing_log(self, project_id: str, timing_log: Dict[str, Any]):
        """
        保存索引耗时日志到文件
        
        Args:
            project_id: 项目ID
            timing_log: 耗时记录
        """
        import json
        from pathlib import Path
        
        # 创建日志目录
        log_dir = Path("logs/indexing")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{project_id}.json"
        
        # 如果文件已存在，追加记录
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 追加新记录
                existing_data['conversations'].append(timing_log)
                existing_data['total_conversations'] = len(existing_data['conversations'])
                
                # 更新统计
                successful_logs = [c for c in existing_data['conversations'] if c.get('status') == 'success']
                if successful_logs:
                    total_times = [c['total_elapsed'] for c in successful_logs]
                    existing_data['total_time'] = round(sum(total_times), 3)
                    existing_data['avg_time'] = round(sum(total_times) / len(total_times), 3)
                
                existing_data['updated_at'] = timing_log.get('completed_at')
                
            except Exception as e:
                print(f"  ⚠️  读取已有日志失败: {e}")
                # 创建新日志
                existing_data = {
                    'project_id': project_id,
                    'created_at': timing_log.get('started_at'),
                    'updated_at': timing_log.get('completed_at'),
                    'total_conversations': 1,
                    'total_time': timing_log.get('total_elapsed', 0),
                    'avg_time': timing_log.get('total_elapsed', 0),
                    'conversations': [timing_log]
                }
        else:
            # 创建新日志文件
            existing_data = {
                'project_id': project_id,
                'created_at': timing_log.get('started_at'),
                'updated_at': timing_log.get('completed_at'),
                'total_conversations': 1,
                'total_time': timing_log.get('total_elapsed', 0),
                'avg_time': timing_log.get('total_elapsed', 0),
                'conversations': [timing_log]
            }
        
        # 写入文件
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 耗时记录已保存到: {log_file}")
        except Exception as e:
            print(f"  ⚠️  保存耗时日志失败: {e}")
    
    def _register_routes(self):
        """注册所有路由"""
        
        @self.app.get("/")
        async def root():
            """根路径"""
            return {
                "service": "GauzRag API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "healthy"}
        
        @self.app.get("/conversation/{conversation_id}/status")
        async def get_conversation_status(conversation_id: int):
            """
            查询 conversation 的索引状态
            
            Args:
                conversation_id: 对话 ID
            
            Returns:
                索引状态信息
            """
            try:
                # 从任意一个 pipeline 获取数据库连接（因为所有项目共享一个数据库）
                if not self.pipelines:
                    # 如果没有任何 pipeline，创建一个临时的
                    from .database import DatabaseManager
                    db = DatabaseManager(
                        host="localhost",
                        port=3306,
                        user="root",
                        password="123456",
                        database="gauz_rag",
                        table="facts"
                    )
                else:
                    # 使用任意一个已存在的 pipeline
                    pipeline = next(iter(self.pipelines.values()))
                    db = pipeline.setup_database()
                
                conversation = db.get_conversation_by_id(conversation_id)
                
                if not conversation:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Conversation {conversation_id} 不存在"
                    )
                
                indexed = conversation['indexed']
                
                return {
                    "conversation_id": conversation_id,
                    "project_id": conversation['project_id'],
                    "indexed": indexed,
                    "status": "completed" if indexed else "indexing",
                    "created_at": conversation['created_at'],
                    "message": "已完成索引（长期记忆）" if indexed else "正在后台索引中（短期记忆）"
                }
            
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        async def _async_indexing_task(
            project_id: str,
            conversation_id: int,
            conversation_text: str,
            metadata: Optional[Dict[str, Any]],
            image_url: Optional[str],
            content_type: Optional[str]
        ):
            """
            后台异步任务：提取 Facts + 构建索引（步骤 1-9）
            
            Args:
                project_id: 项目 ID
                conversation_id: 对话 ID
                conversation_text: 对话文本
                metadata: 元数据
                image_url: 图片URL（如果有）
                content_type: 内容类型
            """
            try:
                import time
                from datetime import datetime
                import json
                
                task_start_time = time.time()
                print(f"\n[后台任务] 开始处理 conversation_id={conversation_id} (开始时间: {time.strftime('%H:%M:%S')})")
                
                # 初始化耗时记录
                timing_log = {
                    'project_id': project_id,
                    'conversation_id': conversation_id,
                    'started_at': datetime.now().isoformat(),
                    'metadata': metadata,
                    'text_length': len(conversation_text),
                    'steps': {}
                }
                
                # 为后台任务创建独立的 Pipeline 实例（避免与前台请求冲突）
                config = GauzRagConfig(
                    project_root=self.base_config.project_root,
                    llm_provider=self.base_config.llm_provider,
                    llm_api_base=self.base_config.llm_api_base,
                    llm_api_key=self.base_config.llm_api_key,
                    llm_model=self.base_config.llm_model,
                    llm_temperature=self.base_config.llm_temperature,
                    llm_max_tokens=self.base_config.llm_max_tokens,
                    embedding_api_key=self.base_config.embedding_api_key,
                    embedding_base_url=self.base_config.embedding_base_url,
                    embedding_model=self.base_config.embedding_model,
                    mysql_host=self.base_config.mysql_host,
                    mysql_port=self.base_config.mysql_port,
                    mysql_user=self.base_config.mysql_user,
                    mysql_password=self.base_config.mysql_password,
                    mysql_database=self.base_config.mysql_database,
                    mysql_table=self.base_config.mysql_table,
                    qdrant_mode=self.base_config.qdrant_mode,
                    qdrant_url=self.base_config.qdrant_url,
                    # 图谱引擎配置
                    graph_engine=self.base_config.graph_engine,
                    lightrag_kv_storage=self.base_config.lightrag_kv_storage,
                    lightrag_vector_storage=self.base_config.lightrag_vector_storage,
                    lightrag_graph_storage=self.base_config.lightrag_graph_storage,
                    neo4j_uri=self.base_config.neo4j_uri,
                    neo4j_user=self.base_config.neo4j_user,
                    neo4j_password=self.base_config.neo4j_password,
                    output_dir=self.base_config.project_root / "output" / project_id,
                    cache_dir=self.base_config.project_root / "cache" / project_id,
                    logs_dir=self.base_config.project_root / "logs" / project_id,
                )
                config.setup_directories()
                
                # 创建独立的 Pipeline 实例
                pipeline = GauzRagPipeline(config, project_id=project_id)
                db = pipeline.setup_database()
                
                # ===== 步骤 1: 提取 Facts 和显性关系（后台，一次性LLM调用）=====
                extract_start = time.time()
                print(f"\n[1/5] 提取 Facts 和显性关系（后台）...")
                
                # 判断是否为图片描述（有 image_url）
                if image_url:
                    # 图片描述模式：直接将 text 作为 fact，不进行 LLM 提取
                    print(f"  - 检测到图片URL，直接使用描述作为 fact")
                    extracted_facts = [{
                        'content': conversation_text,
                        'image_url': image_url
                    }]
                    explicit_relations = []
                    print(f"  ✓ 图片描述作为 fact: {conversation_text[:100]}{'...' if len(conversation_text) > 100 else ''}")
                else:
                    # 普通文本模式：使用 LLM 一次性提取 facts 和显性关系
                    from .fact_extractor import FactExtractor
                    extractor = pipeline.setup_fact_extractor()
                    
                    print(f"  - 调用 LLM 提取（Facts + 显性关系）...")
                    if metadata:
                        print(f"  - 包含 metadata 作为上下文: {list(metadata.keys())}")
                    
                    # 一次性获取 facts 和 relations
                    facts_text, explicit_relations_raw = extractor.extract_from_text(
                        conversation_text, 
                        metadata=metadata
                    )
                    extracted_facts = FactExtractor.parse_facts(facts_text)
                    print(f"  ✓ 提取到 {len(extracted_facts)} 条 facts")
                    
                    if explicit_relations_raw:
                        print(f"  ✓ 提取到 {len(explicit_relations_raw)} 条显性关系 [耗时: {time.time() - extract_start:.2f}秒]")
                    else:
                        print(f"  ✓ 提取到 0 条显性关系 [耗时: {time.time() - extract_start:.2f}秒]")
                
                # 存储到数据库（关联 conversation_id）
                new_count = db.insert_facts(
                    extracted_facts, 
                    conversation_id=conversation_id
                )
                total_facts = db.get_facts_count()
                print(f"  ✓ 已存储，项目共 {total_facts} 条 facts（关联 conversation_id: {conversation_id}）")
                
                # 从数据库获取刚插入的 facts（带 fact_id）
                conv_data = db.get_conversation_with_facts(conversation_id)
                if conv_data and conv_data.get('facts'):
                    new_facts_with_ids = conv_data['facts']
                else:
                    # 兜底方案
                    print(f"  ⚠️  通过 conversation_id 查询失败，使用兜底方案")
                    recent_facts = db.get_recent_facts(limit=new_count * 2)
                    new_facts_with_ids = recent_facts[-new_count:] if len(recent_facts) >= new_count else recent_facts
                
                extract_time = time.time() - extract_start
                print(f"  ✓ Fact提取和存储完成 [耗时: {extract_time:.2f}秒]")
                
                # 记录步骤1耗时
                timing_log['steps']['step1_fact_extraction'] = {
                    'elapsed': round(extract_time, 3),
                    'facts_count': new_count,
                    'total_facts': total_facts
                }
                
                # ===== 步骤 1.5: 处理显性关系（延迟到步骤2后写入Neo4j）=====
                explicit_relations = []
                if not image_url and explicit_relations_raw:
                    print(f"\n  ℹ️  显性关系将在步骤2（Fact节点创建后）写入Neo4j")
                        
                    # 转换为 Neo4j 格式（使用实际的 fact_id）
                    fact_ids = [f["fact_id"] for f in new_facts_with_ids]
                    explicit_relations = FactExtractor.format_relations_for_neo4j(
                        relations=explicit_relations_raw,
                        fact_ids=fact_ids
                    )
                    
                    # 记录步骤1.5耗时
                    timing_log['steps']['step1.5_explicit_relations'] = {
                        'elapsed': round(extract_time, 3),  # 已包含在extract_time中
                        'relations_count': len(explicit_relations)
                    }
                else:
                    # 图片描述或无显性关系
                    timing_log['steps']['step1.5_explicit_relations'] = {
                        'elapsed': 0,
                        'skipped': True
                    }
                
                # ===== 步骤 2: 构建/更新知识图谱 =====
                step2_start = time.time()
                # 使用 LightRAG 引擎
                builder = pipeline.get_graph_builder()
                
                # 检查引擎类型
                from GauzRag.lightrag_builder import GauzRagBuilder
                is_lightrag = isinstance(builder, GauzRagBuilder)
                
                entities_file = pipeline.config.output_dir / "entities.parquet"
                is_incremental = entities_file.exists()
                
                # 用于存储实体映射（内存注入用）
                fact_entities_map = None
                
                # GauzRag 增量更新
                if is_incremental:
                    print(f"\n[2/5] GauzRag 增量更新（新增 {new_count} 条 facts）...")
                else:
                    print(f"\n[2/5] GauzRag 初始化（共 {total_facts} 条 facts）...")
                
                # LightRAG 的 insert 方法（返回实体映射）
                fact_entities_map = await builder.insert_facts(new_facts_with_ids)
                
                step2_time = time.time() - step2_start
                print(f"  ✓ 知识图谱处理完成（项目共 {total_facts} 条 facts）[耗时: {step2_time:.2f}秒]")
                
                # 记录步骤2耗时
                timing_log['steps']['step2_knowledge_graph'] = {
                    'elapsed': round(step2_time, 3),
                    'is_incremental': is_incremental,
                    'total_facts': total_facts
                }
                
                # ===== 步骤 2.5: 写入显性关系到 Neo4j（在Fact节点创建后）=====
                if explicit_relations and hasattr(pipeline.config, 'neo4j_uri'):
                    print(f"\n[2.5/5] 写入显性关系到 Neo4j（{len(explicit_relations)} 条）...")
                    try:
                        from .neo4j_storage import Neo4jGraphStore
                        
                        with Neo4jGraphStore(
                            uri=pipeline.config.neo4j_uri,
                            user=pipeline.config.neo4j_user,
                            password=pipeline.config.neo4j_password,
                            project_id=project_id
                        ) as neo4j_store:
                            success_count = 0
                            for rel in explicit_relations:
                                try:
                                    neo4j_store.add_fact_relation(
                                        source_fact_id=rel['source_fact_id'],
                                        target_fact_id=rel['target_fact_id'],
                                        relation_type=rel['relation_type'],
                                        confidence=rel['confidence'],
                                        metadata={
                                            'source': 'explicit',
                                            'explanation': rel.get('explanation', '')
                                        }
                                    )
                                    success_count += 1
                                except Exception as rel_error:
                                    print(f"  ⚠️ 写入关系失败 (fact {rel['source_fact_id']} -> {rel['target_fact_id']}): {rel_error}")
                        
                        print(f"  ✓ 成功写入 {success_count}/{len(explicit_relations)} 条显性关系到 Neo4j")
                    except Exception as e:
                        print(f"  ⚠️ 写入显性关系失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # ===== 步骤 3: 分析 Fact 语义关系（GauzRag）=====
                # 使用 LLM 分析语义关系（基于 Fact.entities 属性找候选）
                step3_start = time.time()
                print(f"\n[3/5] 使用 LLM 分析 Fact 语义关系（仅分析与历史 Facts 的关系）...")
                try:
                    await pipeline.build_fact_relations_with_lightrag(
                        new_facts_with_ids, 
                        builder,
                        fact_entities_map=fact_entities_map,  # 传递实体映射
                        skip_same_batch=True  # 跳过同批次（已通过显性关系提取完成）
                    )
                    step3_time = time.time() - step3_start
                    print(f"  ✓ Fact 语义关系分析完成 [耗时: {step3_time:.2f}秒]")
                    
                    # 记录步骤3耗时
                    timing_log['steps']['step3_semantic_relations'] = {
                        'elapsed': round(step3_time, 3),
                        'facts_analyzed': len(new_facts_with_ids)
                    }
                except Exception as e:
                    print(f"  ⚠️  Fact 语义关系分析失败: {e}")
                    step3_time = time.time() - step3_start
                    timing_log['steps']['step3_semantic_relations'] = {
                        'elapsed': round(step3_time, 3),
                        'error': str(e)
                    }
                
                # ===== 步骤 3.5: Hybrid Topic 检测（新增）=====
                step35_start = time.time()
                print(f"\n[3.5/5] 使用 Hybrid Topic Detector 为 Facts 分配 Topics...")
                try:
                    from .hybrid_topic_detector import HybridTopicDetector
                    from .vector_store import GauzRagVectorStore
                    from openai import AsyncOpenAI
                    import os
                    
                    # 创建 VectorStore 实例
                    vector_store = GauzRagVectorStore(
                        persist_directory=pipeline.config.output_dir / "qdrant_db",
                        project_id=project_id,
                        use_server=(pipeline.config.qdrant_mode == "server"),
                        server_url=pipeline.config.qdrant_url
                    )
                    
                    # 初始化检测器
                    detector = HybridTopicDetector(
                        vector_store=vector_store,
                        neo4j_store=builder.neo4j_store if hasattr(builder, 'neo4j_store') else None,
                        embedder=pipeline.setup_embedder(),
                        llm_client=AsyncOpenAI(
                            api_key=os.getenv("GAUZ_LLM_API_KEY"),
                            base_url=os.getenv("GAUZ_LLM_API_BASE")
                        ),
                        llm_model=os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini"),
                        project_id=project_id
                    )
                    
                    # 准备 Batch Facts（需要 vector）
                    batch_facts = []
                    embedder = pipeline.setup_embedder()
                    
                    for fact in new_facts_with_ids:
                        # 生成向量（使用 encode 方法，返回 numpy 数组）
                        vector = embedder.encode(fact['content'], convert_to_numpy=True)[0]
                        
                        # 提取实体名称（fact_entities_map 的值是字典列表）
                        raw_entities = fact_entities_map.get(fact['fact_id'], []) if fact_entities_map else []
                        entity_names = [e['name'] if isinstance(e, dict) else e for e in raw_entities]
                        
                        batch_facts.append({
                            "fact_id": fact['fact_id'],
                            "content": fact['content'],
                            "entities": entity_names,  # 传递字符串列表
                            "vector": vector.tolist()  # 转换为 list 以便序列化
                        })
                    
                    if batch_facts:
                        # 处理这个 Batch
                        result = await detector.process_batch(batch_facts)
                        
                        step35_time = time.time() - step35_start
                        print(f"  ✓ Topic 检测完成 [耗时: {step35_time:.2f}秒]")
                        print(f"    - 动作: {result['action']}")
                        if result['topic_id']:
                            print(f"    - Topic ID: {result['topic_id']}")
                            if result.get('details'):
                                print(f"    - 详情: {result['details']}")
                        
                        # 记录步骤3.5耗时
                        timing_log['steps']['step3.5_hybrid_topic'] = {
                            'elapsed': round(step35_time, 3),
                            'action': result['action'],
                            'topic_id': result.get('topic_id'),
                            'facts_count': len(batch_facts)
                        }
                    else:
                        print(f"  ⚠️  无 Facts，跳过 Topic 检测")
                        timing_log['steps']['step3.5_hybrid_topic'] = {
                            'elapsed': 0,
                            'skipped': True
                        }
                    
                    # 关闭 VectorStore
                    vector_store.close()
                        
                except Exception as e:
                    step35_time = time.time() - step35_start
                    print(f"  ⚠️  Topic 检测失败: {e}")
                    import traceback
                    traceback.print_exc()
                    timing_log['steps']['step3.5_hybrid_topic'] = {
                        'elapsed': round(step35_time, 3),
                        'error': str(e)
                    }
                    # 不影响主流程，继续执行
                
                # ===== 步骤 4: 统一构建所有 Embeddings（最后一步，写入 Qdrant）=====
                step4_start = time.time()
                print(f"\n[4/5] 构建 Fact Embeddings 并写入 Qdrant...")
                
                try:
                    embedding_start = time.time()
                    pipeline.build_fact_embeddings(
                        new_facts=new_facts_with_ids,
                        conversation_id=conversation_id,
                        conversation_text=conversation_text,
                        skip_conversation_embedding=True  # 已在中期记忆阶段完成
                    )
                    embedding_time = time.time() - embedding_start
                    print(f"  ✓ {len(new_facts_with_ids)} 条 Facts 已写入 Qdrant [耗时: {embedding_time:.2f}秒]")
                    
                    step4_time = time.time() - step4_start
                    print(f"  ✓ Embeddings 构建完成！[总耗时: {step4_time:.2f}秒]")
                    
                    # 记录步骤4耗时
                    timing_log['steps']['step4_embeddings'] = {
                        'elapsed': round(step4_time, 3),
                        'facts_embedded': len(new_facts_with_ids)
                    }
                
                except Exception as e:
                    print(f"  ⚠️  Embeddings 构建失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    step4_time = time.time() - step4_start
                    timing_log['steps']['step4_embeddings'] = {
                        'elapsed': round(step4_time, 3),
                        'error': str(e)
                    }
                    # 如果 Embedding 失败，不标记为 indexed，保持为短期记忆
                    raise
                
                # ===== 步骤 5: 标记对话为已索引（所有步骤完成）=====
                print(f"\n[5/5] 标记对话为已索引...")
                db.update_conversation_indexed(conversation_id, indexed=True)
                print(f"  ✓ MySQL: conversation_id {conversation_id} 已标记为 indexed=True")
                
                # 同步更新 Qdrant 中的 indexed 状态
                from .vector_store import GauzRagVectorStore
                with GauzRagVectorStore(
                    persist_directory=pipeline.config.output_dir / "qdrant_db",
                    project_id=project_id,
                    use_server=(pipeline.config.qdrant_mode == "server"),
                    server_url=pipeline.config.qdrant_url
                ) as vector_store:
                    vector_store.update_conversation_indexed(conversation_id, indexed=True)
                    print(f"  ✓ Qdrant: conversation_id {conversation_id} 已标记为 indexed=1")
                
                print(f"  ✓ 数据已从短期记忆转为长期记忆")
                
                total_time = time.time() - task_start_time
                print(f"\n[后台任务] 完成！conversation_id={conversation_id} [总耗时: {total_time:.2f}秒 ({time.strftime('%H:%M:%S')})]")
                
                # 完成耗时记录
                timing_log['completed_at'] = datetime.now().isoformat()
                timing_log['total_elapsed'] = round(total_time, 3)
                timing_log['status'] = 'success'
                
                # 保存耗时日志
                self._save_indexing_timing_log(project_id, timing_log)
                
                # 增量更新BM25索引（只添加新的facts）
                print(f"\n[后台任务] 正在更新BM25索引...")
                bm25_start = time.time()
                
                # 准备新facts数据
                bm25_new_facts = [
                    {'fact_id': f['fact_id'], 'content': f['content']}
                    for f in new_facts_with_ids
                ]
                
                bm25_retriever = pipeline.update_bm25_index(bm25_new_facts)
                bm25_elapsed = time.time() - bm25_start
                
                if bm25_retriever:
                    print(f"  ✓ BM25索引已更新并持久化 [耗时: {bm25_elapsed:.2f}秒]")
                else:
                    print(f"  ⚠️  BM25索引更新失败")
                
            except Exception as e:
                import traceback
                error_detail = f"后台任务错误: {str(e)}\n{traceback.format_exc()}"
                print(f"\n[后台任务] 发生错误:\n{error_detail}")
                
                # 记录错误日志
                total_time = time.time() - task_start_time
                timing_log['completed_at'] = datetime.now().isoformat()
                timing_log['total_elapsed'] = round(total_time, 3)
                timing_log['status'] = 'failed'
                timing_log['error'] = str(e)
                self._save_indexing_timing_log(project_id, timing_log)
            finally:
                # 清理异步资源（防止 Event loop closed 错误）
                try:
                    if 'builder' in locals() and builder:
                        await builder.aclose()
                except Exception as e:
                    print(f"  ⚠️  清理 GauzRag Builder 失败: {e}")
        
        @self.app.post("/extract", response_model=dict)
        async def extract_and_rebuild(request: ExtractRequest):
            """
            提取 Facts（短期记忆）并异步构建索引（长期记忆）
            
            模式1：正常模式（replace=False 或未设置）
              - 正常插入新数据
            
            模式2：替换模式（replace=True）
              - 通过 file_hash + chunk_index 查找旧数据
              - 删除旧数据（conversation + facts + 图谱 + embeddings）
              - 插入新数据（正常流程）
            
            同步流程（立即返回，1-2秒）：
            1. [替换模式] 删除旧数据（如果找到）
            2. 保存对话到 conversations 表（indexed=0）
            3. 构建 Conversation Embedding → Qdrant
            4. 返回对话原文（短期记忆）
            
            异步流程（后台任务）：
            1. 提取 Facts 并存储到 facts 表
            2. 构建知识图谱
            3. 生成社区映射
            4. 提取实体映射
            5. 构建 Fact Relations
            6. 生成 Topic Reports
            7. 构建 Fact Embeddings → Qdrant
            8. 标记 conversation 为 indexed=1（数据转为长期记忆）
            
            Args:
                request: 提取请求
            
            Returns:
                对话原文（短期记忆）
            """
            import time
            try:
                print("\n" + "="*80)
                print(f"[Extract] 新请求: project_id={request.project_id}")
                
                pipeline = self._get_pipeline(request.project_id)
                db = pipeline.setup_database()
                
                # ===== 检查是否为替换模式 =====
                old_conversation_id = None
                delete_result = None
                
                if request.replace:
                    print(f"[模式] 🔄 替换模式（replace=True）")
                    
                    # 验证必要的 metadata 字段
                    if not request.metadata or 'file_hash' not in request.metadata or 'chunk_index' not in request.metadata:
                        print("="*80)
                        return {
                            "success": False,
                            "error": "替换模式需要提供 metadata.file_hash 和 metadata.chunk_index"
                        }
                    
                    file_hash = request.metadata['file_hash']
                    chunk_index = request.metadata['chunk_index']
                    
                    print(f"  - 查找旧数据: file_hash={file_hash}, chunk_index={chunk_index}")
                    
                    # 查询旧的 conversation
                    old_conversation_id = db.find_conversation_by_file_chunk(
                        project_id=request.project_id,
                        file_hash=file_hash,
                        chunk_index=chunk_index
                    )
                    
                    if old_conversation_id:
                        print(f"  ✓ 找到旧数据: conversation_id={old_conversation_id}")
                        
                        try:
                            # 删除旧数据（级联删除）
                            delete_result = await self._delete_conversation_cascade(
                                conversation_id=old_conversation_id,
                                project_id=request.project_id,
                                pipeline=pipeline
                            )
                            
                            print(f"  ✓ 旧数据已清理: {delete_result['deleted_facts_count']} 个 facts")
                        except Exception as e:
                            print(f"  ⚠️  删除旧数据失败（将继续插入新数据）: {e}")
                            # 不阻塞新数据插入
                    else:
                        print(f"  ℹ️  未找到旧数据，将作为新数据插入")
                else:
                    print(f"[模式] ➕ 正常模式（replace=False）")
                
                print("="*80)
                
                pipeline = self._get_pipeline(request.project_id)
                db = pipeline.setup_database()
                
                # ===== 步骤 0: 保存对话到 conversations 表（indexed=0）=====
                print(f"\n[0/2] 保存对话原文（短期记忆）...")
                print(f"  - 文本长度: {len(request.text)} 字符")
                
                # 打印输入文本
                print("\n" + "-"*80)
                print("【输入文本】")
                print("-"*80)
                print(request.text[:500] + ("..." if len(request.text) > 500 else ""))
                print("-"*80)
                
                source = request.source_name or "api_input"
                conversation_id = db.insert_conversation(
                    text=request.text,
                    source_file=source,
                    content_type=request.content_type or "conversation",
                    source_metadata=request.metadata
                )
                print(f"  ✓ 已保存对话，conversation_id: {conversation_id}，indexed: False（短期记忆）")
                if request.metadata:
                    print(f"  ✓ Metadata: {request.metadata}")
                
                # ===== 步骤 1: 构建 Conversation Embedding（中期记忆）【同步】=====
                print(f"\n[1/2] 构建 Conversation Embedding（中期记忆）...")
                mid_term_start = time.time()
                try:
                    # 构建 conversation embedding 并写入 Qdrant
                    # 注意：此时 indexed=0（短期记忆），后台任务完成后会更新为 indexed=1
                    conv_metadata = {
                        **(request.metadata or {}),
                        "indexed": 0  # 标记为短期记忆（未完成 Facts 提取）
                    }
                    pipeline.build_conversation_embedding_sync(
                        conversation_id=conversation_id,
                        conversation_text=request.text,
                        project_id=request.project_id,
                        metadata=conv_metadata
                    )
                    mid_term_time = time.time() - mid_term_start
                    print(f"  ✓ Conversation Embedding 已写入 Qdrant（indexed=0，短期记忆）[耗时: {mid_term_time:.2f}秒]")
                except Exception as e:
                    print(f"  ⚠️  构建 Conversation Embedding 失败: {e}")
                    # 不影响主流程，继续执行
                
                # ===== 添加后台任务：异步提取 Facts + 构建索引（使用线程池真正并发）=====
                print(f"\n[2/2] 添加后台任务（提取 Facts + 构建索引）...")
                
                # 使用线程池执行后台任务，不阻塞主线程
                self.executor.submit(
                    lambda: asyncio.run(_async_indexing_task(
                        request.project_id,
                        conversation_id,
                        request.text,
                        request.metadata,
                        request.image_url,
                        request.content_type
                    ))
                )
                print(f"  ✓ 后台任务已添加到线程池（独立线程执行，不阻塞前台）")
                
                print("\n" + "="*80)
                if request.replace and old_conversation_id:
                    print(f"✓ 替换完成！旧数据已删除，新数据已保存")
                    print(f"  - 旧 conversation_id: {old_conversation_id} （已删除）")
                    print(f"  - 新 conversation_id: {conversation_id} （短期记忆）")
                    print(f"  - 删除了 {delete_result['deleted_facts_count']} 个旧 facts")
                else:
                    print(f"✓ 立即返回！对话已保存为短期记忆")
                print(f"  - 短期记忆：Conversation 已保存（可向量检索）")
                print(f"  - 后台任务将提取 Facts 并构建索引（完成后转为长期记忆）")
                print("="*80 + "\n")
                
                # 返回对话原文（短期记忆）
                response_data = {
                    "project_id": request.project_id,
                    "conversation_id": conversation_id,
                    "status": "indexing",
                    "indexed": False,
                    "text": request.text,
                    "message": "对话已保存为短期记忆，后台正在提取 Facts 并构建索引..."
                }
                
                # 如果是替换模式，添加额外信息
                if request.replace and old_conversation_id and delete_result:
                    response_data["replaced"] = {
                        "old_conversation_id": old_conversation_id,
                        "new_conversation_id": conversation_id,
                        "deleted_facts": delete_result['deleted_facts_count'],
                        "delete_details": delete_result['details']
                    
                        }
                
                return response_data
            
            except Exception as e:
                import traceback
                error_detail = f"错误: {str(e)}\n{traceback.format_exc()}"
                print(f"\n[Extract] 发生错误:\n{error_detail}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/search", response_model=BundleQueryResponse)
        async def search(request: QueryRequest):
            """
            三级并行检索 + Bundle聚合
            
            三种检索模式：
            1. 基础语义检索：只提供 query + project_id
            2. 硬过滤检索：使用 filters 字段（时间/元数据过滤）
            3. 扩展检索：使用 expansions 字段（图谱/时序扩展）
            
            Args:
                request: 查询请求
            
            Returns:
                Bundle列表（基于关系网络的聚合结果）
            """
            try:
                # ===== 参数规范化：兼容新旧两种格式 =====
                # 优先使用新格式 (filters/expansions)，如果没有则回退到旧格式
                
                # 1. 处理硬过滤参数
                start_time = None
                end_time = None
                metadata_filter = None
                
                if request.filters:
                    # 新格式
                    if request.filters.time_range:
                        start_time = request.filters.time_range.start
                        end_time = request.filters.time_range.end
                        print(f"[SEARCH] 解析时间过滤: start={start_time}, end={end_time}")
                    metadata_filter = request.filters.metadata
                else:
                    # 旧格式（向后兼容）
                    start_time = request.start_time
                    end_time = request.end_time
                    metadata_filter = request.metadata_filter
                    if start_time or end_time:
                        print(f"[SEARCH] 解析时间过滤（旧格式）: start={start_time}, end={end_time}")
                
                # 2. 处理图谱扩展参数
                use_graph_expansion = False
                max_hops = 2
                graph_relation_types = None
                
                if request.expansions and request.expansions.graph:
                    # 新格式
                    use_graph_expansion = request.expansions.graph.enabled
                    max_hops = request.expansions.graph.max_hops
                    graph_relation_types = request.expansions.graph.relation_types
                elif request.use_graph_expansion is not None:
                    # 旧格式（向后兼容）
                    use_graph_expansion = request.use_graph_expansion
                    max_hops = request.max_hops or 2
                
                # 3. 处理时序扩展参数
                use_temporal_expansion = False
                temporal_mode = "turn"
                temporal_hop_distance = 1
                temporal_direction = "both"
                
                if request.expansions and request.expansions.temporal:
                    # 新格式
                    use_temporal_expansion = request.expansions.temporal.enabled
                    temporal_mode = request.expansions.temporal.mode
                    temporal_hop_distance = request.expansions.temporal.hop_distance
                    temporal_direction = request.expansions.temporal.direction
                elif request.use_temporal_expansion is not None:
                    # 旧格式（向后兼容）
                    use_temporal_expansion = request.use_temporal_expansion
                    temporal_mode = request.temporal_mode or "turn"
                    temporal_hop_distance = request.temporal_hop_distance or 1
                    temporal_direction = request.temporal_direction or "both"
                
                pipeline = self._get_pipeline(request.project_id)
                
                # 初始化数据库管理器（整个搜索流程都会用到）
                db_manager = pipeline.setup_database()
                
                # 检查Qdrant索引（优雅降级）
                from .vector_store import GauzRagVectorStore
                qdrant_dir = pipeline.config.output_dir / "qdrant_db"
                
                # 尝试访问 Qdrant（如果不可用，只返回短期记忆）
                vector_store = None
                conv_results = {'ids': [], 'texts': [], 'scores': [], 'metadatas': []}
                fact_results = {'ids': [], 'contents': [], 'scores': [], 'metadatas': []}
                topic_results = {'ids': [], 'titles': [], 'summaries': [], 'scores': []}
                query_embedding = None
                
                # Server 模式直接尝试连接，本地模式需要检查目录
                should_try_qdrant = (pipeline.config.qdrant_mode == "server") or qdrant_dir.exists()
                
                if should_try_qdrant:
                    try:
                        print(f"[BUNDLE] 尝试访问 Qdrant 长期记忆 ({pipeline.config.qdrant_mode} 模式)...")
                        vector_store = GauzRagVectorStore(
                            qdrant_dir, 
                            request.project_id,
                            use_server=(pipeline.config.qdrant_mode == "server"),
                            server_url=pipeline.config.qdrant_url
                        )
                        embedder = pipeline.setup_embedder()
                        query_embedding = embedder.encode([request.query], convert_to_numpy=True)[0]
                        
                        # ===== 步骤1: 三级并行召回（长期记忆：Qdrant向量索引）=====
                        print(f"[BUNDLE] 三级并行召回 - 长期记忆 (top_k={request.top_k})...")
                        
                        # 1.1 召回 Conversations（长期记忆：indexed=1）
                        # 注意：如果启用，需要添加 indexed=1 过滤
                        conv_filter = metadata_filter.copy() if metadata_filter else {}
                        conv_filter['indexed'] = 0  # 只查询已索引完成的conversations
                        conv_results = vector_store.search_conversations(
                            query_embedding, 
                            top_k=request.top_k, 
                            where=conv_filter,  # 使用带indexed=1过滤的条件
                            start_time=start_time,
                            end_time=end_time
                        )
                        print(f"  - Conversations（长期，indexed=1）: {len(conv_results['ids'])} 条")
                        
                        # 1.2 召回 Facts（长期记忆：indexed=1）
                        if start_time or end_time:
                            print(f"  - 应用时间过滤: {start_time} ~ {end_time}")
                        
                        fact_results = vector_store.search_facts(
                            query_embedding, 
                            top_k=request.top_k * 3 if request.use_bm25 else request.top_k,  # BM25模式多召回一些用于融合
                            where=metadata_filter,
                            start_time=start_time,
                            end_time=end_time
                        )
                        print(f"  - Facts（向量检索）: {len(fact_results['ids'])} 条")
                        
                        # ===== BM25检索（如果启用）=====
                        if request.use_bm25:
                            print(f"[BM25] 启动混合检索...")
                            try:
                                from .bm25_retriever import HybridRetriever
                                
                                # 加载BM25索引（从磁盘或内存）
                                bm25_retriever = pipeline.setup_bm25_retriever(force_rebuild=False)
                                
                                # 如果BM25索引不可用，跳过混合检索
                                if not bm25_retriever:
                                    print(f"  - 警告: BM25索引不可用（项目无facts数据），跳过混合检索")
                                else:
                                    # BM25检索
                                    bm25_results = bm25_retriever.search(request.query, top_k=request.top_k * 3)
                                    print(f"  - Facts（BM25检索）: {len(bm25_results)} 条")
                                    
                                    # 转换向量检索结果格式
                                    vector_results = [
                                        {'id': fact_id, 'score': score}
                                        for fact_id, score in zip(fact_results['ids'], fact_results['scores'])
                                    ]
                                    
                                    # 融合结果
                                    if request.fusion_method == "weighted":
                                        fused_results = HybridRetriever.weighted_score_fusion(
                                            vector_results=vector_results,
                                            bm25_results=bm25_results,
                                            vector_weight=1 - request.bm25_weight,
                                            bm25_weight=request.bm25_weight
                                        )
                                    else:  # rrf
                                        fused_results = HybridRetriever.reciprocal_rank_fusion(
                                            vector_results=vector_results,
                                            bm25_results=bm25_results,
                                            vector_weight=1 - request.bm25_weight,
                                            bm25_weight=request.bm25_weight
                                        )
                                    
                                    # 取Top K
                                    fused_results = fused_results[:request.top_k]
                                    print(f"  - 融合后: {len(fused_results)} 条 (方法: {request.fusion_method})")
                                    
                                    # 更新fact_results
                                    fused_ids = [r['id'] for r in fused_results]
                                    fused_scores = [r['score'] for r in fused_results]
                                    
                                    # 从数据库获取完整信息
                                    fused_facts_data = db_manager.get_facts_by_ids(fused_ids)
                                    fused_contents = [fused_facts_data.get(fid, {}).get('content', '') for fid in fused_ids]
                                    fused_metadatas = [
                                        {
                                            'conversation_id': fused_facts_data.get(fid, {}).get('conversation_id'),
                                            'project_id': fused_facts_data.get(fid, {}).get('project_id'),
                                            'created_at': fused_facts_data.get(fid, {}).get('created_at'),
                                            'created_at_iso': fused_facts_data.get(fid, {}).get('created_at_iso'),
                                            'content_type': fused_facts_data.get(fid, {}).get('content_type'),
                                            'image_url': fused_facts_data.get(fid, {}).get('image_url'),
                                            'retrieval_method': 'hybrid_bm25'
                                        }
                                        for fid in fused_ids
                                    ]
                                    
                                    fact_results = {
                                        'ids': fused_ids,
                                        'scores': fused_scores,
                                        'contents': fused_contents,
                                        'metadatas': fused_metadatas
                                    }
                                
                            except Exception as e:
                                print(f"  ⚠️ BM25检索失败，回退到纯向量检索: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        print(f"  - Facts（最终）: {len(fact_results['ids'])} 条")
                        
                        # 1.3 召回 Topics（长期记忆）- Hybrid Topics
                        try:
                            topic_results_new = await vector_store.search_topics(
                                query_vector=query_embedding,
                                project_id=request.project_id,
                                top_k=min(request.top_k, 5)
                            )
                            print(f"  - Topics（Hybrid检测）: {len(topic_results_new)} 条")
                            
                            # 转换为标准格式
                            topic_results = {
                                'ids': [t['topic_id'] for t in topic_results_new],
                                'scores': [t['score'] for t in topic_results_new],
                                'contents': [],  # 后续从 Neo4j 获取
                                'metadatas': []
                            }
                            
                        except Exception as e:
                            print(f"  ⚠️  Hybrid Topics 搜索失败: {e}")
                            topic_results = {'ids': [], 'scores': [], 'contents': [], 'metadatas': []}
                        
                    except Exception as e:
                        print(f"[BUNDLE] ⚠️ Qdrant 暂时不可用: {str(e)}")
                        print(f"[BUNDLE] 将只返回短期记忆")
                        vector_store = None
                else:
                    print(f"[BUNDLE] Qdrant 索引尚未创建，将只返回短期记忆")
                
                # ===== 步骤1.5: 查询短期记忆（indexed=0 的 conversations，从 Qdrant 向量检索）=====
                print(f"[BUNDLE] 查询短期记忆（Qdrant 检索 indexed=0 的 Conversations）...")
                print(f"  - Query: '{request.query}'")
                if metadata_filter:
                    print(f"  - 应用 metadata 过滤: {metadata_filter}")
                
                # 从 Qdrant 检索 indexed=0 的 Conversations
                short_term_convs = []
                short_term_facts_map = {}
                
                # if vector_store:
                #     try:
                #         # 构建 indexed=0 的过滤条件
                #         short_term_filter = metadata_filter.copy() if metadata_filter else {}
                #         short_term_filter['indexed'] = 0  # 只查询未索引完成的
                        
                #         # 从 Qdrant 检索
                #         short_term_results = vector_store.search_conversations(
                #             query_embedding,
                #             top_k=request.top_k,
                #             where=short_term_filter,
                #             start_time=start_time,
                #             end_time=end_time
                #         )
                        
                #         print(f"  - Qdrant 查到 {len(short_term_results['ids'])} 个 indexed=0 的 conversations")
                #         if short_term_results['ids']:
                #             top_scores = short_term_results['scores'][:3]
                #             print(f"  - Top 3 相似度: {[f'{s:.3f}' for s in top_scores]}")
                        
                #         # 收集短期记忆
                #         for i, conv_id in enumerate(short_term_results['ids']):
                #             short_term_convs.append({
                #                 'conversation_id': conv_id,
                #                 'text': short_term_results['texts'][i],
                #                 'score': float(short_term_results['scores'][i]),
                #                 'metadata': short_term_results['metadatas'][i] if short_term_results['metadatas'] else None
                #             })
                        
                #         # ⚠️ 不再获取facts，因为indexed=0表示后台任务还在处理中，facts可能不完整
                        
                #     except Exception as e:
                #         print(f"  ⚠️ 查询短期记忆失败: {str(e)}")
                #         print(f"  将跳过短期记忆")
                # else:
                #     print(f"  ⚠️ Qdrant 未初始化，跳过短期记忆")
                
                # print(f"  - Conversations（短期，indexed=0）: {len(short_term_convs)} 条")
                
                # ===== 打包短期记忆 =====
                short_memory = None
                # if short_term_convs:
                #     short_memory = ShortTermMemory(
                #         related_memory=None,  # 如果use_refine=True，稍后会填充
                #         conversations=[
                #             {
                #                 'conversation_id': conv['conversation_id'],
                #                 'text': conv['text'],
                #                 'score': conv.get('score', 0.0),
                #                 'metadata': {k: v for k, v in conv.get('metadata', {}).items() 
                #                            if k not in ['conversation_id', 'text', 'score']} 
                #                            if conv.get('metadata') else None
                #             }
                #             for conv in short_term_convs
                #         ],
                #         total_conversations=len(short_term_convs)
                #     )
                #     print(f"[SHORT TERM] 打包短期记忆: {len(short_term_convs)} conversations")
                
                # # ===== 只用长期记忆构建 Bundle =====
                total_convs_long = len(conv_results['ids'])
                total_facts_long = len(fact_results['ids'])
                total_topics_long = len(topic_results['ids'])
                
                if total_convs_long == 0 and total_facts_long == 0 and total_topics_long == 0:
                    print(f"[BUNDLE] 长期记忆未找到结果")
                    if vector_store:
                        vector_store.close()  # 关闭客户端
                    return BundleQueryResponse(
                        query=request.query,
                        project_id=request.project_id,
                        short_term_memory=short_memory,
                        bundles=[],
                        total_bundles=0
                    )
                
                print(f"[BUNDLE] 长期记忆: {total_convs_long} conversations, {total_facts_long} facts, {total_topics_long} topics")
                
                # ===== 步骤1.8: 图谱扩展与语义关系查询（GauzRag 模式）=====
                lightrag_builder = None
                fact_semantic_relations = []  # 存储 Fact 之间的语义关系边
                
                # 在 GauzRag 模式下，总是查询语义关系用于 Bundle 聚合
                if pipeline.config.graph_engine == "lightrag" and fact_results['ids']:
                    try:
                        lightrag_builder = pipeline.get_graph_builder()
                        await lightrag_builder.initialize()
                        
                        # 先查询现有 Facts 之间的语义关系（用于聚合）
                        # 注意：历史数据的关系type可能都是SUPPORT，实际类型存在relation_type属性中
                        print(f"[SEMANTIC RELATIONS] 查询 Facts 之间的语义关系...")
                        async with lightrag_builder.neo4j_store.driver.session() as session:
                            relations_result = await session.run(
                                """
                                MATCH (f1:Fact)-[r]-(f2:Fact)
                                WHERE f1.fact_id IN $fact_ids 
                                  AND f2.fact_id IN $fact_ids
                                  AND f1.fact_id < f2.fact_id
                                  AND f1.project_id = $project_id
                                  AND f2.project_id = $project_id
                                  AND r.relation_type IS NOT NULL
                                RETURN f1.fact_id AS source_id, 
                                       f2.fact_id AS target_id, 
                                       COALESCE(r.relation_type, type(r)) AS relation_type
                                """,
                                fact_ids=fact_results['ids'],
                                project_id=request.project_id
                            )
                            
                            async for record in relations_result:
                                fact_semantic_relations.append({
                                    'source': record['source_id'],
                                    'target': record['target_id'],
                                    'type': record['relation_type']
                                })
                        
                        print(f"  - 找到 {len(fact_semantic_relations)} 条语义关系边")
                    
                    except Exception as e:
                        print(f"  ⚠️  查询语义关系失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # 可选：基于语义关系扩展更多 Facts（8种关系：CAUSE、TEMPORAL、PARALLEL等）
                graph_expansion_data = None  # 新增：存储图谱扩展的详细信息
                if use_graph_expansion and lightrag_builder and fact_results['ids']:
                    print(f"[GRAPH EXPANSION] 基于语义关系扩展 Facts...")
                    print(f"  - 最大跳数: {max_hops}")
                    if graph_relation_types:
                        print(f"  - 指定关系: {', '.join(graph_relation_types)}")
                    else:
                        print(f"  - 使用全部 8 种语义关系")
                    
                    try:
                        # 使用所有召回的 Facts 作为种子（top_k 是几个就用几个）
                        seed_fact_ids = fact_results['ids']
                        print(f"  - 种子 Facts: {len(seed_fact_ids)} 个")
                        
                        expanded_facts = await lightrag_builder.neo4j_store.expand_facts_by_semantic_relations(
                            seed_fact_ids=seed_fact_ids,
                            max_hops=max_hops,
                            relation_types=graph_relation_types
                        )
                        
                        if expanded_facts:
                            print(f"  - 扩展到 {len(expanded_facts)} 个相关 Facts")
                            
                            # 按跳数和关系类型分组统计
                            expansion_by_hops = {}
                            relation_type_counts = {}
                            
                            for expanded_fact in expanded_facts:
                                hop = expanded_fact['hop_distance']
                                if hop not in expansion_by_hops:
                                    expansion_by_hops[hop] = []
                                expansion_by_hops[hop].append(expanded_fact)
                                
                                # 统计关系类型
                                for rel_type in expanded_fact.get('relation_path', []):
                                    relation_type_counts[rel_type] = relation_type_counts.get(rel_type, 0) + 1
                            
                            # 输出统计
                            for hop in sorted(expansion_by_hops.keys()):
                                print(f"    · {hop}跳: {len(expansion_by_hops[hop])} 个 Facts")
                            
                            if relation_type_counts:
                                print(f"    · 关系类型: {', '.join(f'{k}({v})' for k, v in sorted(relation_type_counts.items()))}")
                            
                            # 🔄 新结构：不再平铺扩展的 facts，而是构建 seed -> expanded 的映射
                            # 用于后续在 Bundle 中将扩展的 facts 嵌套到种子 facts 的 hop_facts 字段
                            
                            # Step 1: 先收集每个 (seed, fact_id) 的所有路径，只保留最短的
                            seed_fact_min_hop = {}  # {(seed_id, fact_id): min_hop_level}
                            
                            for expanded_fact in expanded_facts:
                                reasoning_path = expanded_fact['reasoning_path']
                                seed_fact_id = reasoning_path[0]
                                fact_id = expanded_fact['fact_id']
                                hop_level = expanded_fact['hop_distance']
                                
                                key = (seed_fact_id, fact_id)
                                if key not in seed_fact_min_hop:
                                    seed_fact_min_hop[key] = hop_level
                                else:
                                    seed_fact_min_hop[key] = min(seed_fact_min_hop[key], hop_level)
                            
                            # Step 2: 查询所有扩展facts的完整metadata
                            all_expanded_ids = list(set(f['fact_id'] for f in expanded_facts))
                            expanded_facts_metadata = db_manager.get_facts_by_ids(all_expanded_ids) if all_expanded_ids else {}
                            
                            # Step 3: 构建 expansion map，只保留最短路径
                            fact_expansion_map = {}  # {seed_fact_id: {hop_level: [expanded_facts]}}
                            processed_facts = set()  # 已处理的 (seed_id, fact_id, hop_level)
                            
                            for expanded_fact in expanded_facts:
                                reasoning_path = expanded_fact['reasoning_path']
                                relations = expanded_fact.get('relations', [])
                                
                                # 找到种子节点（路径的第一个节点）
                                seed_fact_id = reasoning_path[0]
                                fact_id = expanded_fact['fact_id']
                                hop_level = expanded_fact['hop_distance']
                                
                                # ✅ 只保留最短路径：跳过非最短的路径
                                key = (seed_fact_id, fact_id)
                                if hop_level != seed_fact_min_hop[key]:
                                    continue
                                
                                # ✅ 去重：同一个 (seed, fact, hop) 只处理一次
                                process_key = (seed_fact_id, fact_id, hop_level)
                                if process_key in processed_facts:
                                    continue
                                processed_facts.add(process_key)
                                
                                # 提取语义关系类型
                                relation_types = expanded_fact.get('relation_path', [])
                                if relation_types:
                                    # 使用关系路径中的最后一个关系（直接连接的关系）
                                    relation_type = relation_types[-1]
                                else:
                                    relation_type = "SEMANTIC"
                                
                                # 从数据库获取完整metadata
                                fact_metadata = expanded_facts_metadata.get(fact_id, {})
                                
                                # 构建扩展 fact 的结构（包含完整metadata）
                                expanded_fact_data = {
                                    'fact_id': fact_id,
                                    'content': expanded_fact['content'],
                                    'hop_level': hop_level,
                                    'relation_type': relation_type,  # CAUSE, TEMPORAL, SUPPORT 等
                                    'score': max(0.9 - (hop_level - 1) * 0.1, 0.6),  # 根据跳数调整分数
                                    'path': reasoning_path,
                                    'expanded_from': reasoning_path[-2] if len(reasoning_path) >= 2 else None,
                                    'metadata': {
                                        'conversation_id': fact_metadata.get('conversation_id'),
                                        'project_id': fact_metadata.get('project_id'),
                                        'created_at': fact_metadata.get('created_at'),
                                        'created_at_iso': fact_metadata.get('created_at_iso'),
                                        'content_type': fact_metadata.get('content_type')
                                    }
                                }
                                
                                # 初始化种子节点的映射
                                if seed_fact_id not in fact_expansion_map:
                                    fact_expansion_map[seed_fact_id] = {}
                                
                                # 按跳数分组
                                hop_key = f"{hop_level}hop" if hop_level == 1 else f"{hop_level}hops"
                                if hop_key not in fact_expansion_map[seed_fact_id]:
                                    fact_expansion_map[seed_fact_id][hop_key] = []
                                
                                fact_expansion_map[seed_fact_id][hop_key].append(expanded_fact_data)
                            
                            # 将 expansion_map 存储到结果中（后续在 Bundle 构建时使用）
                            fact_results['expansion_map'] = fact_expansion_map
                            
                            # 统计去重后的数量（全局唯一 fact_id）
                            unique_expanded_facts = set()
                            deduplicated_by_hop = {}
                            
                            for seed_map in fact_expansion_map.values():
                                for hop_key, hop_facts in seed_map.items():
                                    for fact in hop_facts:
                                        unique_expanded_facts.add(fact['fact_id'])
                                        # 按跳数统计（注意：同一 fact 可能在不同种子的同一跳出现，这里会重复计数）
                                        if hop_key not in deduplicated_by_hop:
                                            deduplicated_by_hop[hop_key] = set()
                                        deduplicated_by_hop[hop_key].add(fact['fact_id'])
                            
                            # 转换 set 为 count
                            hop_counts = {k: len(v) for k, v in deduplicated_by_hop.items()}
                            
                            print(f"  ✓ 图谱扩展映射完成：{len(fact_expansion_map)} 个种子节点")
                            print(f"  ✓ 原始扩展: {len(expanded_facts)} 条路径，去重后: {len(unique_expanded_facts)} 个唯一 Facts")
                            for hop_key in sorted(hop_counts.keys()):
                                print(f"    · {hop_key}: {hop_counts[hop_key]} 个唯一 Facts")
                            
                            # ❌ 已禁用：不将扩展的Facts添加到主列表，避免重复
                            # 扩展的Facts只通过种子Fact的hop_facts字段展示
                            # expanded_fact_ids = list(unique_expanded_facts)
                            # if expanded_fact_ids:
                            #     # 合并到主列表
                            #     original_count = len(fact_results['ids'])
                            #     
                            #     # 从数据库获取扩展Facts的完整信息
                            #     expanded_facts_data = db_manager.get_facts_by_ids(expanded_fact_ids)
                            #     
                            #     # 同步更新所有字段
                            #     for exp_id in expanded_fact_ids:
                            #         if exp_id not in fact_results['ids']:  # 避免重复
                            #             fact_results['ids'].append(exp_id)
                            #             fact_data = expanded_facts_data.get(exp_id, {})
                            #             fact_results['contents'].append(fact_data.get('content', ''))
                            #             fact_results['scores'].append(0.5)  # 扩展的Facts给固定分数
                            #             fact_results['metadatas'].append({
                            #                 'source': 'graph_expansion',
                            #                 'conversation_id': fact_data.get('conversation_id')
                            #             })
                            #     
                            #     print(f"  ✓ 已将扩展的Facts添加到主列表: {original_count} → {len(fact_results['ids'])} 个")
                            
                            print(f"  ✓ 扩展的Facts保留在hop_facts字段中，不添加到主列表（避免重复）")
                            
                            # 🔄 简化版的图谱扩展统计（详细信息在 facts 的 hop_facts 字段中）
                            
                            graph_expansion_data = {
                                'enabled': True,
                                'seed_facts_count': len(fact_expansion_map),
                                'total_expanded_facts': len(unique_expanded_facts),
                                'max_hops_configured': max_hops,
                                'actual_hops_reached': max(expansion_by_hops.keys()) if expansion_by_hops else 0,
                                'by_hop_level': hop_counts
                            }
                            
                            print(f"  ✓ 图谱扩展完成，总 Facts: {len(fact_results['ids'])} 个")
                            
                            # 重新查询语义关系（包含扩展的 Facts）
                            fact_semantic_relations = []
                            async with lightrag_builder.neo4j_store.driver.session() as session:
                                relations_result = await session.run(
                                    """
                                    MATCH (f1:Fact)-[r]-(f2:Fact)
                                    WHERE f1.fact_id IN $fact_ids 
                                      AND f2.fact_id IN $fact_ids
                                      AND f1.fact_id < f2.fact_id
                                      AND f1.project_id = $project_id
                                      AND f2.project_id = $project_id
                                      AND r.relation_type IS NOT NULL
                                    RETURN f1.fact_id AS source_id, 
                                           f2.fact_id AS target_id, 
                                           COALESCE(r.relation_type, type(r)) AS relation_type
                                    """,
                                    fact_ids=fact_results['ids'],
                                    project_id=request.project_id
                                )
                                
                                async for record in relations_result:
                                    fact_semantic_relations.append({
                                        'source': record['source_id'],
                                        'target': record['target_id'],
                                        'type': record['relation_type']
                                    })
                            
                            print(f"  - 更新后找到 {len(fact_semantic_relations)} 条语义关系边")
                        else:
                            print(f"  - 未找到可扩展的 Facts")
                    
                    except Exception as e:
                        print(f"  ⚠️  图谱扩展失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # 新增：时序多跳扩展（基于时间/轮次）
                temporal_expansion_data = None
                if use_temporal_expansion and fact_results['ids']:
                    print(f"[TEMPORAL EXPANSION] 时序多跳扩展（{temporal_mode} 模式）...")
                    try:
                        # 使用所有召回的 Facts 作为种子（top_k 是几个就用几个）
                        seed_fact_ids = fact_results['ids']
                        print(f"  - 种子 Facts: {len(seed_fact_ids)} 个")
                        
                        if temporal_mode == "turn":
                            # 基于轮次扩展
                            expansion_result = db_manager.expand_facts_by_turn(
                                seed_fact_ids=seed_fact_ids,
                                hop_distance=temporal_hop_distance,
                                direction=temporal_direction
                            )
                            print(f"  - 轮次范围: {expansion_result['turn_range']}")
                        elif temporal_mode == "time":
                            # 基于时间窗口扩展
                            time_window = temporal_hop_distance * 5  # 1跳=5分钟
                            expansion_result = db_manager.expand_facts_by_time(
                                seed_fact_ids=seed_fact_ids,
                                time_window_minutes=time_window,
                                direction=temporal_direction
                            )
                            print(f"  - 时间范围: {expansion_result['time_range']}")
                        else:
                            raise ValueError(f"Invalid temporal_mode: {temporal_mode}")
                        
                        expanded_facts = expansion_result['facts']
                        expanded_conversations = expansion_result['conversations']
                        
                        if expanded_facts:
                            print(f"  - 扩展到 {len(expanded_facts)} 个 Facts")
                            print(f"  - 涉及 {len(expanded_conversations)} 个对话")
                            
                            # 合并到已有的 fact_results（去重）
                            existing_fact_ids = set(fact_results['ids'])
                            new_facts_count = 0
                            
                            for fact in expanded_facts:
                                if fact['fact_id'] not in existing_fact_ids:
                                    fact_results['ids'].append(fact['fact_id'])
                                    fact_results['contents'].append(fact['content'])
                                    fact_results['scores'].append(0.6)  # 时序扩展固定分数
                                    fact_results['metadatas'].append({
                                        'conversation_id': fact['conversation_id'],
                                        'image_url': fact.get('image_url'),
                                        'temporal_expanded': True,
                                        'expansion_mode': request.temporal_mode
                                    })
                                    existing_fact_ids.add(fact['fact_id'])
                                    new_facts_count += 1
                            
                            print(f"  ✓ 时序扩展完成：新增 {new_facts_count} 个 Facts")
                            
                            temporal_expansion_data = {
                                'enabled': True,
                                'mode': temporal_mode,
                                'hop_distance': temporal_hop_distance,
                                'direction': temporal_direction,
                                'seed_facts_count': len(seed_fact_ids),
                                'total_expanded_facts': len(expanded_facts),
                                'new_facts_added': new_facts_count,
                                'conversations_count': len(expanded_conversations),
                                'range': expansion_result.get('turn_range') or expansion_result.get('time_range')
                            }
                        else:
                            print(f"  - 未找到可扩展的 Facts")
                    
                    except Exception as e:
                        print(f"  ⚠️  时序扩展失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # ===== 步骤2: 构建关系图 =====
                print(f"[BUNDLE] 构建关系图...")
                import json
                
                # 2.1 获取Facts的详细信息（包含conversation_id）- 只用长期记忆
                fact_ids = fact_results['ids']
                facts_data = db_manager.get_facts_by_ids(fact_ids) if fact_ids else {}
                
                print(f"  - 从数据库获取 {len(facts_data)} 条 Facts（长期记忆）")
                
                # 2.2 读取 Topic 信息和 Fact -> Topic 映射
                fact_to_topics = {}
                topic_reports_map = {}
                
                # 从 Neo4j 批量读取 Hybrid Topics
                try:
                    if lightrag_builder and topic_results and topic_results.get('ids'):
                        topic_ids = topic_results['ids']
                        
                        if topic_ids:
                            # 从 Neo4j 批量读取 Topics
                            async with lightrag_builder.neo4j_store.driver.session() as session:
                                neo4j_result = await session.run(
                                    """
                                    MATCH (t:Topic)
                                    WHERE t.project_id = $project_id 
                                      AND t.topic_id IN $topic_ids
                                    OPTIONAL MATCH (f:Fact)-[:BELONGS_TO]->(t)
                                    RETURN t.topic_id AS topic_id,
                                           t.title AS title,
                                           t.summary AS summary,
                                           collect(f.fact_id) AS fact_ids
                                    """,
                                    project_id=request.project_id,
                                    topic_ids=topic_ids
                                )
                                
                                async for record in neo4j_result:
                                    tid = record['topic_id']
                                    topic_reports_map[tid] = {
                                        'report': {
                                            'title': record['title'] or f'Topic {tid}',
                                            'summary': record['summary'] or ''
                                        }
                                    }
                                    
                                    # 构建 fact -> topic 映射
                                    for fid in record['fact_ids']:
                                        if fid:
                                            if fid not in fact_to_topics:
                                                fact_to_topics[fid] = []
                                            fact_to_topics[fid].append(tid)
                            
                            print(f"  - 从 Neo4j 加载 {len(topic_reports_map)} 个 Topics")
                except Exception as e:
                    print(f"  ⚠️  加载 Topics 失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 2.3 构建关系映射
                # fact_id -> conversation_id
                fact_to_conv = {}
                for fid, fdata in facts_data.items():
                    conv_id = fdata.get('conversation_id')
                    if conv_id:
                        fact_to_conv[fid] = conv_id
                
                print(f"  - Fact->Conv 关系: {len(fact_to_conv)} 条")
                print(f"  - Fact->Topic 关系: {sum(len(v) for v in fact_to_topics.values())} 条")
                
                # ===== 步骤3: Union-Find 连通分量检测 =====
                print(f"[BUNDLE] 使用 Union-Find 检测连通分量...")
                
                # 为每个元素分配唯一ID（用字符串表示类型）
                all_elements = set()
                
                # 添加长期记忆的conversations
                for conv_id in conv_results['ids']:
                    all_elements.add(('conv', conv_id, 'long_term'))
                
                # ⚠️ 不添加短期记忆的conversations到bundle（已在short_term_memory中单独展示）
                # for conv_data in short_term_convs:
                #     all_elements.add(('conv', conv_data['conversation_id'], 'short_term'))
                
                # 添加长期记忆的facts
                for fact_id in fact_results['ids']:
                    all_elements.add(('fact', fact_id, 'long_term'))
                
                # 添加短期记忆的facts
                for fact_id in short_term_facts_map.keys():
                    if fact_id not in fact_results['ids']:  # 避免重复
                        all_elements.add(('fact', fact_id, 'short_term'))
                
                # 添加长期记忆的topics（短期记忆没有topics）
                for topic_id in topic_results['ids']:
                    all_elements.add(('topic', topic_id, 'long_term'))
                
                print(f"  - 总元素数: {len(all_elements)} (长期+短期)")
                
                # Union-Find 数据结构
                parent = {elem: elem for elem in all_elements}
                
                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])  # 路径压缩
                    return parent[x]
                
                def union(x, y):
                    root_x = find(x)
                    root_y = find(y)
                    if root_x != root_y:
                        parent[root_x] = root_y
                
                # 建立连接
                fact_conv_edges = 0
                fact_topic_edges = 0
                fact_fact_edges = 0
                
                # fact -> conversation（处理长期和短期）
                for fid, conv_id in fact_to_conv.items():
                    # 寻找fact元素（可能是长期或短期）
                    fact_elem_long = ('fact', fid, 'long_term')
                    fact_elem_short = ('fact', fid, 'short_term')
                    fact_elem = None
                    
                    if fact_elem_long in all_elements:
                        fact_elem = fact_elem_long
                    elif fact_elem_short in all_elements:
                        fact_elem = fact_elem_short
                    
                    if fact_elem:
                        # 寻找conversation元素（只连接长期记忆的conversations）
                        conv_elem_long = ('conv', conv_id, 'long_term')
                        
                        if conv_elem_long in all_elements:
                            union(fact_elem, conv_elem_long)
                            fact_conv_edges += 1
                        # ⚠️ 不再连接短期记忆的conversations（已在short_term_memory中单独展示）
                
                # fact -> topic（只有长期记忆的facts有topics）
                for fid, topic_ids in fact_to_topics.items():
                    fact_elem = ('fact', fid, 'long_term')
                    if fact_elem in all_elements:
                        for topic_id in topic_ids:
                            topic_elem = ('topic', topic_id, 'long_term')
                            if topic_elem in all_elements:
                                union(fact_elem, topic_elem)
                                fact_topic_edges += 1
                
                # fact -> fact（基于语义关系，来自图谱扩展）
                if fact_semantic_relations:
                    for relation in fact_semantic_relations:
                        fact1_id = relation['source']
                        fact2_id = relation['target']
                        
                        # 查找两个 fact 元素
                        fact1_elem = ('fact', fact1_id, 'long_term') if ('fact', fact1_id, 'long_term') in all_elements else None
                        fact2_elem = ('fact', fact2_id, 'long_term') if ('fact', fact2_id, 'long_term') in all_elements else None
                        
                        if fact1_elem and fact2_elem:
                            union(fact1_elem, fact2_elem)
                            fact_fact_edges += 1
                
                total_edges = fact_conv_edges + fact_topic_edges + fact_fact_edges
                if fact_fact_edges > 0:
                    print(f"  - 建立 {total_edges} 条边 (Fact-Conv: {fact_conv_edges}, Fact-Topic: {fact_topic_edges}, Fact-Fact语义: {fact_fact_edges})")
                else:
                    print(f"  - 建立 {total_edges} 条边")
                
                # 分组连通分量
                components = {}
                for elem in all_elements:
                    root = find(elem)
                    if root not in components:
                        components[root] = []
                    components[root].append(elem)
                
                print(f"  - 检测到 {len(components)} 个 Bundle")
                
                # ===== 步骤4: 构建Bundle响应 =====
                print(f"[BUNDLE] 构建响应...")
                
                bundles = []
                for bundle_idx, (root, elements) in enumerate(components.items()):
                    # 分类元素
                    bundle_convs = []
                    bundle_facts = []
                    bundle_topics = []
                    
                    for elem in elements:
                        elem_type, elem_id, source = elem  # 解包：类型、ID、来源
                        
                        if elem_type == 'conv':
                            if source == 'long_term':
                                # 长期记忆的conversation
                                idx = conv_results['ids'].index(elem_id)
                                # 清理 metadata：移除与顶层字段重复的数据、以及 None 值
                                clean_metadata = {k: v for k, v in conv_results['metadatas'][idx].items() 
                                                if k not in ['conversation_id', 'text', 'source_identifier']
                                                and v is not None}
                                bundle_convs.append(ConversationItem(
                                    conversation_id=elem_id,
                                    text=conv_results['texts'][idx],
                                    score=conv_results['scores'][idx],
                                    metadata=clean_metadata if clean_metadata else None
                                ))
                            else:  # short_term
                                # 短期记忆的conversation
                                conv_data = next((c for c in short_term_convs if c['conversation_id'] == elem_id), None)
                                if conv_data:
                                    # 清理 metadata：移除与顶层字段重复的数据、以及 None 值
                                    raw_metadata = conv_data.get('metadata', {})
                                    clean_metadata = {k: v for k, v in raw_metadata.items() 
                                                    if k not in ['conversation_id', 'text', 'source_identifier']
                                                    and v is not None} if raw_metadata else {}
                                    bundle_convs.append(ConversationItem(
                                        conversation_id=elem_id,
                                        text=conv_data['text'],
                                        score=conv_data['score'],
                                        metadata=clean_metadata if clean_metadata else None
                                    ))
                        
                        elif elem_type == 'fact':
                            if source == 'long_term':
                                # 长期记忆的fact
                                fact_data = facts_data.get(elem_id, {})
                                idx = fact_results['ids'].index(elem_id)
                                # 清理 metadata：移除与顶层字段重复的数据、图扩展技术字段、以及 None 值
                                clean_metadata = {k: v for k, v in fact_results['metadatas'][idx].items() 
                                                if k not in ['fact_id', 'content', 'image_url', 'source', 'hop_level', 'source_identifier']
                                                and v is not None}
                                
                                # 🔄 检查是否有图谱扩展的子节点
                                expansion_map = fact_results.get('expansion_map', {})
                                hop_facts = expansion_map.get(elem_id, None)
                                
                                bundle_facts.append(FactItem(
                                    fact_id=elem_id,
                                    content=fact_data.get('content', ''),
                                    score=fact_results['scores'][idx],
                                    image_url=fact_data.get('image_url'),
                                    hop_facts=hop_facts,  # 🔄 添加多跳扩展的 facts
                                    metadata=clean_metadata if clean_metadata else None
                                ))
                            else:  # short_term
                                # 短期记忆的fact
                                fact_data = short_term_facts_map.get(elem_id, {})
                                # 清理 metadata：移除与顶层字段重复的数据、图扩展技术字段、以及 None 值
                                raw_metadata = fact_data.get('metadata', {})
                                clean_metadata = {k: v for k, v in raw_metadata.items() 
                                                if k not in ['fact_id', 'content', 'image_url', 'source', 'hop_level', 'source_identifier']
                                                and v is not None} if raw_metadata else {}
                                bundle_facts.append(FactItem(
                                    fact_id=elem_id,
                                    content=fact_data.get('content', ''),
                                    score=0.5,  # 短期记忆固定分数
                                    image_url=fact_data.get('image_url'),
                                    metadata=clean_metadata if clean_metadata else None
                                ))
                        
                        elif elem_type == 'topic':
                            # Topics 只有长期记忆
                            idx = topic_results['ids'].index(elem_id)
                            report_data = topic_reports_map.get(elem_id, {})
                            report = report_data.get('report', {})
                            bundle_topics.append(TopicItem(
                                topic_id=elem_id,
                                title=report.get('title', f'Topic {elem_id}'),
                                summary=report.get('summary', ''),
                                score=topic_results['scores'][idx]
                            ))
                    
                    # 按score排序
                    bundle_convs.sort(key=lambda x: x.score, reverse=True)
                    bundle_facts.sort(key=lambda x: x.score, reverse=True)
                    bundle_topics.sort(key=lambda x: x.score, reverse=True)
                    
                    bundles.append(BundleResponse(
                        bundle_id=bundle_idx,
                        conversations=bundle_convs,
                        facts=bundle_facts,
                        topics=bundle_topics
                    ))
                
                # 按Bundle的最高score排序
                def get_bundle_max_score(bundle):
                    scores = []
                    if bundle.conversations:
                        scores.extend([c.score for c in bundle.conversations])
                    if bundle.facts:
                        scores.extend([f.score for f in bundle.facts])
                    if bundle.topics:
                        scores.extend([t.score for t in bundle.topics])
                    return max(scores) if scores else 0
                
                bundles.sort(key=get_bundle_max_score, reverse=True)
                
                # 重新分配bundle_id
                for i, bundle in enumerate(bundles):
                    bundle.bundle_id = i
                
                print(f"[BUNDLE] 返回 {len(bundles)} 个 Bundles")
                for i, b in enumerate(bundles[:3]):  # 打印前3个
                    print(f"  Bundle {i}: {len(b.conversations)} convs, {len(b.facts)} facts, {len(b.topics)} topics")
                
                # ===== 步骤4.5: 查询最近N轮对话（多轮对话上下文）=====
                recent_turns_bundle = None
                recent_turns_limit = 5  # 默认查询最近5轮
                
                # print(f"\n[RECENT TURNS] 查询最近 {recent_turns_limit} 轮对话...")
                # try:
                #     # 从数据库查询最近N轮对话（按 turn 倒序）
                #     recent_convs_result = db_manager.get_recent_conversations(
                #         project_id=request.project_id,
                #         limit=recent_turns_limit,
                #         metadata_filter=request.metadata_filter
                #     )
                    
                #     if recent_convs_result:
                #         print(f"  - 找到 {len(recent_convs_result)} 轮对话")
                        
                #         # 构建 recent_turns_bundle
                #         recent_conversations = []
                #         for conv in recent_convs_result:
                #             # 清理 metadata：移除与顶层字段重复的数据、以及 None 值
                #             raw_metadata = conv.get('metadata', {})
                #             clean_metadata = {k: v for k, v in raw_metadata.items() 
                #                             if k not in ['conversation_id', 'text', 'source_identifier']
                #                             and v is not None} if raw_metadata else {}
                            
                #             recent_conversations.append(ConversationItem(
                #                 conversation_id=conv['conversation_id'],
                #                 text=conv['text'],
                #                 score=1.0,  # 最近对话给满分
                #                 metadata=clean_metadata if clean_metadata else None
                #             ))
                        
                #         recent_turns_bundle = RecentTurnsBundle(
                #             conversations=recent_conversations
                #         )
                        
                #         print(f"  - 构建 recent_turns bundle: {len(recent_conversations)} 条对话")
                #     else:
                #         print(f"  - 未找到最近对话")
                
                # except Exception as e:
                #     print(f"  ⚠️  查询最近对话失败: {str(e)}")
                #     import traceback
                #     traceback.print_exc()
                
                # ===== 步骤5: LLM精炼（可选）=====
                if request.use_refine:
                    import asyncio
                    import time
                    
                    # 计算总任务数
                    total_tasks = len(bundles)
                    if recent_turns_bundle and recent_turns_bundle.conversations:
                        total_tasks += 1
                    if short_memory and short_memory.conversations:
                        total_tasks += 1
                    
                    print(f"\n[REFINE] 使用LLM并行精炼 {total_tasks} 个任务（{len(bundles)} bundles + recent_turns + short_term）...")
                    start_time = time.time()
                    
                    # 定义异步精炼任务
                    async def refine_bundle_async(bundle):
                        # 在线程池中执行同步的LLM调用
                        loop = asyncio.get_event_loop()
                        refined_data = await loop.run_in_executor(
                            None,
                            pipeline.refine_bundle_with_llm,
                            request.query,
                            bundle
                        )
                        return ('bundle', refined_data, bundle)
                    
                    async def refine_recent_turns_async(recent_turns):
                        # 在线程池中执行同步的LLM调用
                        loop = asyncio.get_event_loop()
                        refined_data = await loop.run_in_executor(
                            None,
                            pipeline.refine_recent_turns_with_llm,
                            request.query,
                            recent_turns
                        )
                        return ('recent_turns', refined_data)
                    
                    async def refine_short_term_async(short_term):
                        # 在线程池中执行同步的LLM调用
                        loop = asyncio.get_event_loop()
                        refined_data = await loop.run_in_executor(
                            None,
                            pipeline.refine_short_term_with_llm,
                            request.query,
                            short_term
                        )
                        return ('short_term', refined_data)
                    
                    # 并行执行所有精炼任务（bundles + recent_turns + short_term）
                    tasks = [refine_bundle_async(bundle) for bundle in bundles]
                    if recent_turns_bundle and recent_turns_bundle.conversations:
                        tasks.append(refine_recent_turns_async(recent_turns_bundle))
                    if short_memory and short_memory.conversations:
                        tasks.append(refine_short_term_async(short_memory))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 构造响应
                    refined_bundles = []
                    for result in results:
                        if isinstance(result, Exception):
                            print(f"  ⚠️  精炼异常: {str(result)}")
                            continue
                        
                        task_type = result[0]
                        if task_type == 'bundle':
                            _, refined_data, bundle = result
                            refined_bundles.append(RefinedBundleResponse(
                                bundle_id=bundle.bundle_id,
                                related_memory=refined_data["related_memory"],
                                quote=refined_data.get("quote"),
                                conversations=bundle.conversations,
                                facts=bundle.facts,
                                topics=bundle.topics
                            ))
                        elif task_type == 'recent_turns':
                            _, refined_data = result
                            recent_turns_bundle.related_memory = refined_data
                        elif task_type == 'short_term':
                            _, refined_data = result
                            short_memory.related_memory = refined_data
                    
                    elapsed = time.time() - start_time
                    print(f"  ✓ 完成精炼（并行处理，耗时 {elapsed:.2f}秒）")
                    
                    if vector_store:
                        vector_store.close()  # 关闭客户端
                    return BundleQueryResponse(
                        query=request.query,
                        project_id=request.project_id,
                        short_term_memory=short_memory,
                        recent_turns=recent_turns_bundle,
                        bundles=refined_bundles,
                        total_bundles=len(refined_bundles),
                        refined=True,
                        graph_expansion=graph_expansion_data,  # 添加图谱扩展信息
                        temporal_expansion=temporal_expansion_data  # 添加时序扩展信息
                    )
                else:
                    if vector_store:
                        vector_store.close()  # 关闭客户端
                    return BundleQueryResponse(
                        query=request.query,
                        project_id=request.project_id,
                        short_term_memory=short_memory,
                        recent_turns=recent_turns_bundle,
                        bundles=bundles,
                        total_bundles=len(bundles),
                        refined=False,
                        graph_expansion=graph_expansion_data,  # 添加图谱扩展信息
                        temporal_expansion=temporal_expansion_data  # 添加时序扩展信息
                    )
            
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/agenticSearch", response_model=AgenticSearchResponse)
        async def agentic_search(request: AgenticSearchRequest):
            """
            Agentic 搜索：使用 LLM 将自然语言转化为结构化查询
            
            工作流程：
            1. 使用 LLM 分析自然语言查询
            2. 提取关键信息：时间、用户、意图等
            3. 构建结构化的 /search 参数
            4. 执行实际搜索并返回结果
            
            示例输入：
            - "帮我找一下用户A在12月关于工作的对话"
            - "为什么Melanie最近很忙？需要完整的因果链"
            - "用户上周提到孩子后说了什么？"
            
            Args:
                request: Agentic搜索请求
            
            Returns:
                包含意图理解、结构化查询和实际结果的响应
            """
            try:
                import json
                from openai import OpenAI
                pipeline = self._get_pipeline(request.project_id)
                
                # 初始化 LLM 客户端
                llm_client = OpenAI(
                    api_key=pipeline.config.llm_api_key,
                    base_url=pipeline.config.llm_api_base
                )
                
                # 构建 prompt
                system_prompt = """你是一个智能查询分析助手。你的任务是将用户的自然语言查询转化为结构化的搜索参数。

**三种检索模式**：
1. **基础语义检索**：最简单，只需要提取查询意图
2. **硬过滤检索**：需要精确的时间/用户/会话/轮次过滤
3. **扩展检索**：需要因果推理、上下文扩展

**分析步骤**：
1. 识别查询意图（查找、分析、推理等）
2. 提取时间信息（如"12月"、"上周"、"最近"）
3. 提取元数据过滤条件（用户名、会话、轮次等）
4. 判断是否需要图谱扩展（"为什么"、"原因"、"因果链"）
   - 如果需要因果推理，使用 CAUSE 关系
   - 如果需要时间线，使用 TEMPORAL 关系
   - 如果需要证据支撑，使用 SUPPORT 关系
   - 如果没有明确要求，使用全部关系（null）
5. 判断是否需要时序扩展（"前后"、"上下文"、"后来"）

**8种语义关系类型**：
- CAUSE: 因果关系（"为什么"、"原因"、"导致"）
- TEMPORAL: 时间顺序（"之后"、"接着"、"时间线"）
- SUPPORT: 支持/补充（"证据"、"支持"、"补充"）
- ELABORATE: 详细阐述（"详细"、"具体"、"展开"）
- CONTRADICT: 矛盾冲突（"矛盾"、"相反"、"冲突"）
- ANALOGY: 类比关系（"类似"、"像"、"对比"）
- CONDITIONAL: 条件关系（"如果"、"条件"、"假设"）
- PARALLEL: 并列关系（"或"、"和"、"选项"、"并列"）

**输出 JSON 格式**：
{
  "intent": "查询意图描述",
  "search_type": "basic" | "filtered" | "expanded" | "combined",
  "filters": {
    "time_range": {"start": "ISO8601", "end": "ISO8601"},
    "metadata": {"username": "...", "turn": 5}
  },
  "expansions": {
    "graph": {
      "enabled": true, 
      "max_hops": 2,
      "relation_types": ["CAUSE", "TEMPORAL"]  // 指定关系，或 null 表示全部
    },
    "temporal": {"enabled": true, "mode": "turn", "hop_distance": 1, "direction": "both"}
  }
}

**注意**：
- 相对时间需要根据 context.current_date 计算绝对时间
- 如果没有明确的过滤/扩展需求，对应字段为 null
- 轮次过滤放在 filters.metadata.turn
- max_hops 默认 2（最多2跳）
- relation_types 只在有明确要求时指定，否则为 null（使用全部关系）
"""
                
                # 准备用户消息
                context_info = request.context or {}
                user_message = f"""自然语言查询：「{request.query}」

上下文信息：
{json.dumps(context_info, ensure_ascii=False, indent=2) if context_info else "无"}

请分析并输出结构化查询参数（JSON 格式）。"""
                
                print(f"[AGENTIC SEARCH] 分析查询: {request.query}")
                
                # 调用 LLM
                response = llm_client.chat.completions.create(
                    model=pipeline.config.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,  # 低温度，更稳定
                    response_format={"type": "json_object"}
                )
                
                # 解析 LLM 响应
                llm_output = json.loads(response.choices[0].message.content)
                print(f"[AGENTIC SEARCH] LLM 分析结果: {json.dumps(llm_output, ensure_ascii=False, indent=2)}")
                
                # 构建结构化查询
                structured_query = QueryRequest(
                    query=request.query,
                    project_id=request.project_id,
                    top_k=request.top_k,
                    filters=SearchFilters(
                        time_range=TimeRangeFilter(
                            start=llm_output.get("filters", {}).get("time_range", {}).get("start"),
                            end=llm_output.get("filters", {}).get("time_range", {}).get("end")
                        ) if llm_output.get("filters", {}).get("time_range") else None,
                        metadata=llm_output.get("filters", {}).get("metadata")
                    ) if llm_output.get("filters") else None,
                    expansions=SearchExpansions(
                        graph=GraphExpansionConfig(
                            enabled=True,  # 如果graph对象存在且enabled=True，才会到这里
                            max_hops=llm_output.get("expansions", {}).get("graph", {}).get("max_hops") or 2,
                            relation_types=llm_output.get("expansions", {}).get("graph", {}).get("relation_types")
                        ) if (
                            llm_output.get("expansions", {}).get("graph") and 
                            llm_output.get("expansions", {}).get("graph", {}).get("enabled")
                        ) else None,
                        temporal=TemporalExpansionConfig(
                            enabled=True,  # 如果temporal对象存在且enabled=True，才会到这里
                            mode=llm_output.get("expansions", {}).get("temporal", {}).get("mode") or "turn",
                            hop_distance=llm_output.get("expansions", {}).get("temporal", {}).get("hop_distance") or 1,
                            direction=llm_output.get("expansions", {}).get("temporal", {}).get("direction") or "both"
                        ) if (
                            llm_output.get("expansions", {}).get("temporal") and 
                            llm_output.get("expansions", {}).get("temporal", {}).get("enabled")
                        ) else None
                    ) if llm_output.get("expansions") else None
                )
                
                print(f"[AGENTIC SEARCH] 结构化查询:")
                print(f"  - 意图: {llm_output.get('intent', '未知')}")
                print(f"  - 类型: {llm_output.get('search_type', 'basic')}")
                if structured_query.filters:
                    print(f"  - 硬过滤: 已启用")
                    if structured_query.filters.time_range:
                        print(f"    · 时间范围: {structured_query.filters.time_range.start} ~ {structured_query.filters.time_range.end}")
                    if structured_query.filters.metadata:
                        print(f"    · 元数据: {structured_query.filters.metadata}")
                if structured_query.expansions:
                    print(f"  - 扩展: 已启用")
                    if structured_query.expansions.graph and structured_query.expansions.graph.enabled:
                        print(f"    · 图谱: max_hops={structured_query.expansions.graph.max_hops}", end="")
                        if structured_query.expansions.graph.relation_types:
                            print(f", relations={', '.join(structured_query.expansions.graph.relation_types)}")
                        else:
                            print()
                    if structured_query.expansions.temporal and structured_query.expansions.temporal.enabled:
                        print(f"    · 时序: mode={structured_query.expansions.temporal.mode}, direction={structured_query.expansions.temporal.direction}")
                
                # 执行实际搜索（调用内部 search 函数）
                print(f"[AGENTIC SEARCH] 执行结构化搜索...")
                search_results = await search(structured_query)
                
                # 返回结果
                return AgenticSearchResponse(
                    original_query=request.query,
                    interpreted_intent=llm_output.get("intent", "查询相关信息"),
                    structured_query=structured_query,
                    search_results=search_results
                    )
            
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/search/facts", response_model=FactQueryResponse)
        async def search_facts(request: FactQueryRequest):
            """
            Fact 级别的查询召回（支持 LLM Refine）
            
            返回内容：
            - refine=False（默认）: 返回结构化的 Facts 列表
            - refine=True: 只返回 LLM 整合后的自然语言回答
            
            Args:
                request: Fact 查询请求
                  - refine: 是否使用 LLM 整合召回的 facts 成自然语言回答
            
            Returns:
                - refine=False: results 包含 Facts 列表，refined_answer 为 None
                - refine=True: results 为 None，refined_answer 包含 LLM 回答
            """
            try:
                pipeline = self._get_pipeline(request.project_id)
                
                # ===== 查询短期记忆（未索引的 conversations 及其 facts）=====
                print(f"[Search Facts] 查询短期记忆...")
                if request.metadata_filter:
                    print(f"  - 应用 metadata 过滤: {request.metadata_filter}")
                db = pipeline.setup_database()
                short_term_data = db.get_unindexed_conversations_with_facts(
                    project_id=request.project_id,
                    metadata_filter=request.metadata_filter
                )
                
                # 收集短期记忆的 facts（简单全文匹配）
                short_term_facts = []
                for conv in short_term_data:
                    for fact in conv['facts']:
                        # 简单的关键词匹配（可以后续优化为向量相似度）
                        if request.query.lower() in fact['content'].lower():
                            # 清理 metadata：移除与 FactSearchResult 顶层字段重复的数据、图扩展技术字段、以及 None 值
                            raw_metadata = conv.get('metadata', {})
                            clean_metadata = {k: v for k, v in raw_metadata.items() 
                                            if k not in ['fact_id', 'content', 'relevance_score', 'community_id', 
                                                        'community_name', 'community_report', 'conversation_id', 
                                                        'source', 'hop_level', 'source_identifier']
                                            and v is not None} if raw_metadata else None
                            short_term_facts.append({
                                'fact_id': fact['fact_id'],
                                'content': fact['content'],
                                'relevance_score': 0.5,  # 给短期记忆固定中等分数
                                'source': 'short_term_memory',
                                'conversation_id': conv['conversation_id'],
                                'metadata': clean_metadata
                            })
                
                print(f"  - 短期记忆匹配: {len(short_term_facts)} 条 facts")
                
                # ===== 查询长期记忆（Qdrant 向量索引，优雅降级）=====
                results = []
                qdrant_dir = pipeline.config.output_dir / "qdrant_db"
                
                if qdrant_dir.exists():
                    try:
                        print(f"[Search Facts] 查询长期记忆（Qdrant）...")
                        if request.metadata_filter:
                            print(f"  - 使用 metadata 过滤: {request.metadata_filter}")
                        results = pipeline.search_facts(
                            query=request.query,
                            top_k=request.top_k,
                            include_community=request.include_community,
                            metadata_filter=request.metadata_filter
                        )
                        
                        # 标记长期记忆的来源
                        for r in results:
                            r['source'] = 'long_term_memory'
                        
                        print(f"  - 长期记忆匹配: {len(results)} 条 facts")
                        
                    except Exception as e:
                        print(f"[Search Facts] ⚠️ Qdrant 暂时不可用（可能正在构建索引）: {str(e)}")
                        print(f"[Search Facts] 将只返回短期记忆")
                        results = []
                else:
                    print(f"[Search Facts] Qdrant 索引尚未创建，将只返回短期记忆")
                
                # ===== 合并短期和长期记忆 =====
                # 短期记忆在前（优先展示），长期记忆在后
                all_results = short_term_facts + results
                
                # 去重（按 fact_id）
                seen_fact_ids = set()
                unique_results = []
                for r in all_results:
                    if r['fact_id'] not in seen_fact_ids:
                        seen_fact_ids.add(r['fact_id'])
                        unique_results.append(r)
                
                # 限制最终返回数量
                final_results = unique_results[:request.top_k]
                
                print(f"\n[Search] 查询: {request.query}")
                print(f"  - 短期记忆: {len(short_term_facts)} 条")
                print(f"  - 长期记忆: {len(results)} 条")
                print(f"  - 合并后: {len(final_results)} 条")
                
                # 转换为响应格式
                fact_results = []
                for result in final_results:
                    # 清理 metadata：移除与顶层字段重复的数据、图扩展技术字段、以及 None 值
                    raw_metadata = result.get('metadata', {})
                    clean_metadata = {k: v for k, v in raw_metadata.items() 
                                    if k not in ['fact_id', 'content', 'relevance_score', 'community_id', 
                                                'community_name', 'community_report', 'source', 'hop_level', 'source_identifier']
                                    and v is not None} if raw_metadata else None
                    fact_results.append(FactSearchResult(
                        fact_id=result['fact_id'],
                        content=result['content'],
                        relevance_score=result['relevance_score'],
                        community_id=result.get('community_id'),
                        community_name=result.get('community_name'),
                        community_report=result.get('community_report'),
                        metadata=clean_metadata
                    ))
                
                # 根据 refine 参数决定返回内容
                if request.refine:
                    # refine=True: 只返回 LLM 整合后的自然语言回答
                    if fact_results:
                        refined_answer = self._refine_facts_with_llm(
                            query=request.query,
                            facts=fact_results
                        )
                    else:
                        refined_answer = "抱歉，未检索到相关信息。"
                    
                    return FactQueryResponse(
                        query=request.query,
                        project_id=request.project_id,
                        results=None,  # 不返回原始 facts
                        total_facts=len(fact_results),
                        refined_answer=refined_answer
                    )
                else:
                    # refine=False: 返回结构化的 Facts 列表
                    return FactQueryResponse(
                        query=request.query,
                        project_id=request.project_id,
                        results=fact_results,
                        total_facts=len(fact_results),
                        refined_answer=None
                    )
            
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/fact/{project_id}/{fact_id}/relations", response_model=FactRelationResponse)
        async def get_fact_relations(project_id: str, fact_id: int):
            """
            获取指定 Fact 的所有关系
            
            Args:
                project_id: 项目 ID
                fact_id: Fact ID
            
            Returns:
                Fact 的关系信息
            """
            try:
                pipeline = self._get_pipeline(project_id)
                
                # 从 Neo4j 获取 Fact Relations（新架构）
                if pipeline.config.graph_engine == "lightrag":
                    # 使用 Neo4j
                    lightrag_builder = pipeline.get_graph_builder()
                    
                    # 异步获取关系
                    import asyncio
                    relations_list = asyncio.run(
                        lightrag_builder.neo4j_store.get_fact_relations(fact_id, max_relations=100)
                    )
                    
                    # 获取 Fact 内容
                    db = pipeline.setup_database()
                    facts_dict = db.get_facts_by_ids([fact_id])
                    fact_content = facts_dict.get(fact_id, {}).get('content', f"Fact {fact_id}")
                    
                    # 格式化为 API 响应（Neo4j 是无向图，不区分 outgoing/incoming）
                    return FactRelationResponse(
                        fact_id=fact_id,
                        content=fact_content,
                        outgoing_relations=relations_list,  # Neo4j 中是无向的
                        incoming_relations=[],  # 无向图，合并到 outgoing
                        total_relations=len(relations_list)
                    )
                else:
                    # 兼容旧的 JSON 方式（GraphRAG）
                    fact_relations_path = pipeline.config.output_dir / "fact_relations.json"
                    if not fact_relations_path.exists():
                        raise HTTPException(
                            status_code=400,
                            detail=f"项目 {project_id} 尚未构建 Fact Relations"
                        )
                    
                    graph_builder = pipeline.setup_fact_relation_graph()
                    relations = graph_builder.get_fact_relations(fact_id)
                    
                    db = pipeline.setup_database()
                    facts_dict = db.get_facts_by_ids([fact_id])
                    fact_content = facts_dict.get(fact_id, {}).get('content', f"Fact {fact_id}")
                    
                    return FactRelationResponse(
                        fact_id=fact_id,
                        content=fact_content,
                        outgoing_relations=relations['outgoing'],
                        incoming_relations=relations['incoming'],
                        total_relations=relations['total']
                    )
            
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/search/time_dimension", response_model=TimeDimensionQueryResponse)
        async def search_time_dimension(request: TimeDimensionQueryRequest):
            """
            时间维度的混合检索（整合图检索 + 向量检索）
            
            Args:
                request: 时间维度查询请求
            
            Returns:
                时间维度查询响应
            
            示例请求:
                POST /search/time_dimension
                {
                    "query": "GPT-4的最新进展",
                    "project_id": "my_project",
                    "start_time": "2024-01-01T00:00:00",
                    "end_time": "2024-12-31T23:59:59",
                    "top_k": 10,
                    "use_graph": true,
                    "use_vector": true,
                    "entity_filter": "GPT-4"
                }
            """
            try:
                # 获取 Pipeline
                pipeline = self._get_pipeline(request.project_id)
                
                # 调用时间维度检索
                results = pipeline.search_with_time_dimension(
                    query=request.query,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    top_k=request.top_k,
                    use_graph=request.use_graph,
                    use_vector=request.use_vector,
                    entity_filter=request.entity_filter,
                    metadata_filter=request.metadata_filter
                )
                
                return TimeDimensionQueryResponse(
                    query=request.query,
                    project_id=request.project_id,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    graph_results_count=len(results['graph_results']),
                    vector_results_count=len(results['vector_results']),
                    merged_results=results['merged_results'],
                    total_results=len(results['merged_results']),
                    time_distribution=results.get('time_distribution')
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/project/{project_id}/cache")
        async def clear_project_cache(project_id: str):
            """
            清除项目缓存，强制重新加载 ChromaDB 索引
            
            Args:
                project_id: 项目 ID
            
            Returns:
                操作结果
            """
            if project_id in self.pipelines:
                del self.pipelines[project_id]
                return {"message": f"Project {project_id} cache cleared successfully"}
            return {"message": f"Project {project_id} not in cache"}
        
        @self.app.get("/project/{project_id}/status", response_model=ProjectStatusResponse)
        async def project_status(project_id: str):
            """
            获取项目状态
            
            Args:
                project_id: 项目 ID
            
            Returns:
                项目状态
            """
            try:
                pipeline = self._get_pipeline(project_id)
                
                # 检查 facts 数量
                db = pipeline.setup_database()
                facts_count = db.get_facts_count()
                
                # 检查是否有 embeddings
                embeddings_path = pipeline.config.output_dir / "community_embeddings.pkl"
                has_embeddings = embeddings_path.exists()
                
                # 检查是否有 community mapping
                community_facts_path = pipeline.config.output_dir / "community_facts.json"
                has_community_mapping = community_facts_path.exists()
                
                return ProjectStatusResponse(
                    project_id=project_id,
                    facts_count=facts_count,
                    has_embeddings=has_embeddings,
                    has_community_mapping=has_community_mapping,
                    ready_for_search=has_embeddings and has_community_mapping
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8006, reload: bool = False):
        """
        运行服务
        
        Args:
            host: 主机地址
            port: 端口
            reload: 是否自动重载
        """
        uvicorn.run(self.app, host=host, port=port, reload=reload)


def create_app(config_path: Optional[Path] = None) -> FastAPI:
    """
    创建 FastAPI 应用（工厂函数）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        FastAPI 应用实例
    """
    import os
    
    # 加载配置
    if config_path is None:
        # 尝试从环境变量获取项目根目录
        project_root = os.environ.get("GAUZRAG_PROJECT_ROOT")
        if project_root:
            config_path = Path(project_root) / ".env"
        else:
            config_path = Path.cwd() / ".env"
    
    config = GauzRagConfig.from_env(config_path)
    
    # 设置项目根目录
    if config.project_root == Path.cwd():
        project_root = os.environ.get("GAUZRAG_PROJECT_ROOT")
        if project_root:
            config.project_root = Path(project_root)
    
    if not config.validate():
        raise RuntimeError("配置验证失败")
    
    # 创建 API
    api = GauzRagAPI(config)
    return api.app

