"""
GauzRag 主流程管道
整合所有模块，提供端到端的处理流程
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from .config import GauzRagConfig
from .database import DatabaseManager
from .fact_extractor import FactExtractor
# GraphBuilder 延迟导入（避免 scipy 依赖问题）
from .community_mapper import CommunityMapper
from .embedder import DashScopeEmbedder
from .searcher import CommunitySearcher, EmbeddingIndexBuilder, FactSearcher
from .vector_store import GauzRagVectorStore
from .fact_relation_builder import (
    EntityCommunityMapper,
    FactCommunityLocator,
    FactRelationAnalyzer,
    FactRelationGraphBuilder
)
from .entity_extractor import FactEntityExtractor
from .bm25_retriever import BM25Retriever, HybridRetriever
import asyncio

# 动态导入 GauzRag Builder（仅在需要时）
try:
    from .lightrag_graph_builder import GauzRagGraphBuilder as GauzRagBuilder
    from .lightrag_graph_builder import GauzRagEntityMapper
    GAUZRAG_AVAILABLE = True
    print(f"✓ GauzRag 模块导入成功: {GauzRagBuilder}")
except ImportError as e:
    print(f"✗ GauzRag 导入失败 (ImportError): {e}")
    GAUZRAG_AVAILABLE = False
    GauzRagBuilder = None
    GauzRagEntityMapper = None
except Exception as e:
    print(f"✗ GauzRag 导入失败 (其他异常): {type(e).__name__}: {e}")
    GAUZRAG_AVAILABLE = False
    GauzRagBuilder = None
    GauzRagEntityMapper = None


class GauzRagPipeline:
    """GauzRag 完整流程管道"""
    
    def __init__(self, config: GauzRagConfig, project_id: str = None):
        """
        初始化 Pipeline
        
        Args:
            config: GauzRag 配置
            project_id: 项目 ID（用于数据库中的数据隔离）
        """
        self.config = config
        self.project_id = project_id
        self.config.setup_directories()
        
        # 初始化各模块
        self.db_manager = None
        self.fact_extractor = None
        self.graph_builder = None
        self.lightrag_builder = None  # GauzRag 构建器
        self.community_mapper = None
        self.embedder = None
        self.searcher = None
        self.fact_searcher = None
        self.entity_community_mapper = None
        self.fact_relation_analyzer = None
        self.fact_relation_graph = None
        self.fact_entity_extractor = None
        self.bm25_retriever = None  # BM25检索器
    
    def setup_database(self) -> DatabaseManager:
        """初始化数据库管理器"""
        if self.db_manager is None:
            self.db_manager = DatabaseManager(
                host=self.config.mysql_host,
                port=self.config.mysql_port,
                user=self.config.mysql_user,
                password=self.config.mysql_password,
                database=self.config.mysql_database,
                table=self.config.mysql_table,
                project_id=self.project_id  # 传入 project_id
            )
            self.db_manager.create_facts_table()
        return self.db_manager
    
    def setup_fact_extractor(self) -> FactExtractor:
        """初始化 Facts 提取器"""
        if self.fact_extractor is None:
            self.fact_extractor = FactExtractor(
                api_base=self.config.llm_api_base,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        return self.fact_extractor
    
    def setup_graph_builder(self):
        """初始化图谱构建器（固定使用 GauzRag）"""
        return self.setup_lightrag_builder()
    
    def setup_lightrag_builder(self) -> 'GauzRagBuilder':
        """初始化 GauzRag 构建器"""
        if not GAUZRAG_AVAILABLE:
            raise RuntimeError(
                "GauzRag 不可用。请检查 GauzRag 是否正确安装。"
            )
        
        if self.lightrag_builder is None:
            import os
            
            # 获取存储配置
            graph_storage = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
            vector_storage = os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
            kv_storage = os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
            
            # 构建工作目录
            working_dir = self.config.output_dir / self.project_id if self.project_id else self.config.output_dir
            
            self.lightrag_builder = GauzRagBuilder(
                working_dir=working_dir,
                llm_api_key=self.config.llm_api_key,
                llm_api_base=self.config.llm_api_base,
                llm_model=self.config.llm_model,
                embedding_model=self.config.embedding_model,
                embedding_dim=1024,  # text-embedding-v4 的维度
                # 存储配置
                graph_storage=graph_storage,
                vector_storage=vector_storage,
                kv_storage=kv_storage,
                # Neo4j 配置
                neo4j_uri=self.config.neo4j_uri if hasattr(self.config, 'neo4j_uri') else None,
                neo4j_user=self.config.neo4j_user if hasattr(self.config, 'neo4j_user') else None,
                neo4j_password=self.config.neo4j_password if hasattr(self.config, 'neo4j_password') else None,
                project_id=self.project_id or "default",  # ← 添加 project_id
                # Qdrant 配置
                qdrant_url=self.config.qdrant_url,
            )
            
            print(f"✓ GauzRag Builder 已初始化")
            print(f"  - 图存储: {graph_storage}")
            print(f"  - 向量存储: {vector_storage}")
            print(f"  - KV存储: {kv_storage}")
        
        return self.lightrag_builder
    
    def get_graph_builder(self) -> 'GauzRagBuilder':
        """
        获取图谱构建器（固定使用 GauzRag）
        
        Returns:
            GauzRagBuilder
        """
        return self.setup_lightrag_builder()
    
    def setup_community_mapper(self) -> CommunityMapper:
        """初始化 Community 映射器"""
        if self.community_mapper is None:
            self.community_mapper = CommunityMapper(
                output_dir=self.config.output_dir
            )
        return self.community_mapper
    
    def setup_embedder(self) -> DashScopeEmbedder:
        """初始化 Embedder"""
        if self.embedder is None:
            self.embedder = DashScopeEmbedder(
                api_key=self.config.embedding_api_key,
                base_url=self.config.embedding_base_url,
                model=self.config.embedding_model
            )
        return self.embedder
    
    def setup_searcher(self) -> CommunitySearcher:
        """初始化搜索器"""
        embeddings_path = self.config.output_dir / "community_embeddings.pkl"
        community_facts_path = self.config.output_dir / "community_facts.json"
        
        if not embeddings_path.exists() or not community_facts_path.exists():
            raise RuntimeError(
                "Embedding 索引或 Community Facts 文件不存在（已废弃的离线构建方式）。"
            )
        
        if self.searcher is None:
            self.searcher = CommunitySearcher(embeddings_path, community_facts_path)
        return self.searcher
    
    def setup_fact_searcher(self) -> FactSearcher:
        """
        初始化 Fact 搜索器（已弃用）
        
        注意：此方法使用旧的 pickle 方案，推荐使用 search_facts_chromadb()（现基于 Qdrant）
        """
        fact_embeddings_path = self.config.output_dir / "fact_embeddings.pkl"
        community_facts_path = self.config.output_dir / "community_facts.json"
        fact_community_reports_path = self.config.output_dir / "fact_community_reports.json"
        
        if not fact_embeddings_path.exists() or not community_facts_path.exists():
            raise RuntimeError(
                "Fact Embedding 索引或 Community Facts 文件不存在（已废弃的离线构建方式）。"
            )
        
        if self.fact_searcher is None:
            self.fact_searcher = FactSearcher(
                fact_embeddings_path, 
                community_facts_path,
                fact_community_reports_path if fact_community_reports_path.exists() else None
            )
        return self.fact_searcher
    
    def setup_bm25_retriever(self, force_rebuild: bool = False) -> BM25Retriever:
        """
        初始化或加载BM25检索器
        
        Args:
            force_rebuild: 是否强制重建索引（默认False，优先加载已有索引）
        
        Returns:
            BM25Retriever实例
        """
        # 如果已在内存中，直接返回
        if self.bm25_retriever is not None and not force_rebuild:
            return self.bm25_retriever
        
        # BM25索引文件路径
        working_dir = self.config.output_dir / self.project_id if self.project_id else self.config.output_dir
        bm25_index_path = working_dir / "bm25_index.pkl"
        
        # 优先加载已有索引
        if not force_rebuild and bm25_index_path.exists():
            try:
                print(f"[BM25] 加载已有索引: {bm25_index_path}")
                self.bm25_retriever = BM25Retriever.load(str(bm25_index_path))
                return self.bm25_retriever
            except Exception as e:
                print(f"[BM25] 加载索引失败，将重新构建: {e}")
        
        # 构建新索引
        print(f"[BM25] 正在构建索引...")
        import time
        start_time = time.time()
        
        # 从数据库读取所有已索引的facts
        db_manager = self.setup_database()
        all_facts = db_manager.get_all_facts(project_id=self.project_id)
        
        if not all_facts:
            print(f"[BM25] 警告: 项目 {self.project_id} 没有facts数据")
            return None
        
        # 构建BM25索引
        corpus = [{'id': f['fact_id'], 'text': f['content']} for f in all_facts]
        self.bm25_retriever = BM25Retriever(corpus)
        
        # 持久化索引
        try:
            self.bm25_retriever.save(str(bm25_index_path))
        except Exception as e:
            print(f"[BM25] 警告: 索引保存失败: {e}")
        
        elapsed = time.time() - start_time
        print(f"[BM25] 索引构建完成: {len(corpus)} 条facts [耗时: {elapsed:.2f}秒]")
        
        return self.bm25_retriever
    
    def update_bm25_index(self, new_facts: List[Dict[str, Any]]) -> BM25Retriever:
        """
        增量更新BM25索引（追加新facts）
        
        Args:
            new_facts: 新增的facts列表 [{'fact_id': int, 'content': str}, ...]
        
        Returns:
            更新后的BM25Retriever实例
        """
        if not new_facts:
            print(f"[BM25] 没有新facts，跳过索引更新")
            return self.bm25_retriever
        
        print(f"[BM25] 增量更新索引: {len(new_facts)} 条新facts")
        import time
        start_time = time.time()
        
        # 加载或初始化索引
        if self.bm25_retriever is None:
            self.setup_bm25_retriever(force_rebuild=False)
        
        # 如果索引为空，直接构建
        if self.bm25_retriever is None:
            corpus = [{'id': f['fact_id'], 'text': f['content']} for f in new_facts]
            self.bm25_retriever = BM25Retriever(corpus)
        else:
            # 增量添加新文档
            new_docs = [{'id': f['fact_id'], 'text': f['content']} for f in new_facts]
            self.bm25_retriever.add_documents(new_docs)
        
        # 持久化更新后的索引
        working_dir = self.config.output_dir / self.project_id if self.project_id else self.config.output_dir
        bm25_index_path = working_dir / "bm25_index.pkl"
        
        try:
            self.bm25_retriever.save(str(bm25_index_path))
        except Exception as e:
            print(f"[BM25] 警告: 索引保存失败: {e}")
        
        elapsed = time.time() - start_time
        print(f"[BM25] 索引更新完成 [耗时: {elapsed:.2f}秒, 总计: {len(self.bm25_retriever.corpus)} 条facts]")
        
        return self.bm25_retriever
    
    def search_facts_chromadb(
        self,
        query: str,
        top_k: int = 10,
        search_mode: str = "fact",
        metadata_filter: dict = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        基于 Qdrant 的 Fact 检索（推荐，支持时间维度）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            search_mode: "fact" | "conversation" | "hybrid"
            metadata_filter: 元数据过滤条件
            start_time: 开始时间（ISO 格式），如 "2024-01-01T00:00:00"
            end_time: 结束时间（ISO 格式），如 "2024-12-31T23:59:59"
        
        Returns:
            [
                {
                    'fact_id': int,
                    'content': str,
                    'score': float,
                    'conversation_id': int,
                    'created_at': str  # 时间戳
                },
                ...
            ]
        """
        embedder = self.setup_embedder()
        
        # 使用 with 语句确保 VectorStore 正确关闭
        with GauzRagVectorStore(
            persist_directory=self.config.output_dir / "qdrant_db",
            project_id=self.project_id,
            use_server=(self.config.qdrant_mode == "server"),
            server_url=self.config.qdrant_url
        ) as vector_store:
            # 生成查询向量
            query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
            
            # 执行检索
            if search_mode == "fact":
                raw_results = vector_store.search_facts(
                    query_embedding, 
                    top_k, 
                    where=metadata_filter,
                    start_time=start_time,
                    end_time=end_time
                )
                # 构建返回结果（包含 metadata）
                results = []
                for i, fid in enumerate(raw_results['ids']):
                    results.append({
                        'fact_id': fid,
                        'content': raw_results['contents'][i],
                        'score': 1 - raw_results['distances'][i],
                        'conversation_id': raw_results['metadatas'][i].get('conversation_id'),
                        'metadata': raw_results['metadatas'][i]  # ← 完整 metadata
                    })
                return results
            elif search_mode == "conversation":
                # 先搜索对话，再返回对话内的 facts
                conv_results = vector_store.search_conversations(
                    query_embedding, 
                    top_k=3, 
                    where=metadata_filter,
                    start_time=start_time,
                    end_time=end_time
                )
                all_facts = []
                for conv_id in conv_results['ids']:
                    # 合并 conversation_id 过滤和用户的 metadata_filter
                    combined_filter = {"conversation_id": conv_id}
                    if metadata_filter:
                        combined_filter.update(metadata_filter)
                    fact_results = vector_store.search_facts(
                        query_embedding,
                        top_k=100,
                        where=combined_filter,
                        start_time=start_time,
                        end_time=end_time
                    )
                    for i, fid in enumerate(fact_results['ids']):
                        all_facts.append({
                            'fact_id': fid,
                            'content': fact_results['contents'][i],
                            'score': 1 - fact_results['distances'][i],
                            'conversation_id': fact_results['metadatas'][i].get('conversation_id'),
                            'metadata': fact_results['metadatas'][i]  # ← 添加 metadata
                        })
                all_facts.sort(key=lambda x: x['score'], reverse=True)
                return all_facts[:top_k]
            elif search_mode == "hybrid":
                # 三级混合检索
                hybrid_results = vector_store.hybrid_search(
                    query_embedding,
                    top_k_facts=top_k,
                    top_k_conversations=3,
                    top_k_communities=2
                )
                
                # 合并去重
                fact_ids_seen = set()
                merged_facts = []
                
                # Fact 级别
                for i, fid in enumerate(hybrid_results['facts']['ids']):
                    if fid not in fact_ids_seen:
                        fact_ids_seen.add(fid)
                        merged_facts.append({
                            'fact_id': fid,
                            'content': hybrid_results['facts']['contents'][i],
                            'score': 1 - hybrid_results['facts']['distances'][i],
                            'conversation_id': hybrid_results['facts']['metadatas'][i].get('conversation_id'),
                            'source': 'fact_search',
                            'metadata': hybrid_results['facts']['metadatas'][i]  # ← 添加 metadata
                        })
                
                # Conversation 级别
                for conv_id in hybrid_results['conversations']['ids']:
                    conv_facts = vector_store.search_facts(
                        query_embedding,
                        top_k=5,
                        where={"conversation_id": conv_id}
                    )
                    for i, fid in enumerate(conv_facts['ids']):
                        if fid not in fact_ids_seen:
                            fact_ids_seen.add(fid)
                            merged_facts.append({
                                'fact_id': fid,
                                'content': conv_facts['contents'][i],
                                'score': 1 - conv_facts['distances'][i],
                                'conversation_id': conv_facts['metadatas'][i].get('conversation_id'),
                                'source': 'conversation_search',
                                'metadata': conv_facts['metadatas'][i]  # ← 添加 metadata
                            })
                
                merged_facts.sort(key=lambda x: x['score'], reverse=True)
                return merged_facts[:top_k]
            else:
                raise ValueError(f"未知的 search_mode: {search_mode}")
    
    def setup_entity_community_mapper(self) -> EntityCommunityMapper:
        """初始化实体-社区映射器"""
        if self.entity_community_mapper is None:
            self.entity_community_mapper = EntityCommunityMapper(self.config.output_dir)
        return self.entity_community_mapper
    
    def setup_fact_relation_analyzer(self) -> FactRelationAnalyzer:
        """初始化 Fact 关系分析器"""
        if self.fact_relation_analyzer is None:
            self.fact_relation_analyzer = FactRelationAnalyzer(
                api_base=self.config.llm_api_base,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature
            )
        return self.fact_relation_analyzer
    
    def setup_fact_relation_graph(self) -> FactRelationGraphBuilder:
        """初始化 Fact 关系图构建器"""
        if self.fact_relation_graph is None:
            self.fact_relation_graph = FactRelationGraphBuilder(self.config.output_dir)
            # 尝试加载现有图
            graph_path = self.config.output_dir / "fact_relations.json"
            if graph_path.exists():
                self.fact_relation_graph.load_graph(graph_path)
        return self.fact_relation_graph
    
    def setup_fact_entity_extractor(self) -> FactEntityExtractor:
        """初始化 Fact 实体提取器"""
        if self.fact_entity_extractor is None:
            self.fact_entity_extractor = FactEntityExtractor(self.config.output_dir)
        return self.fact_entity_extractor
    
    def search_with_time_dimension(
        self,
        query: str,
        start_time: str,
        end_time: str,
        top_k: int = 10,
        use_graph: bool = True,
        use_vector: bool = True,
        entity_filter: Optional[str] = None,
        metadata_filter: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        时间维度的混合检索（整合图检索 + 向量检索）
        
        Args:
            query: 用户查询
            start_time: 开始时间（ISO 格式），如 "2024-01-01T00:00:00"
            end_time: 结束时间（ISO 格式），如 "2024-12-31T23:59:59"
            top_k: 每种检索方式返回的数量
            use_graph: 是否使用 Neo4j 图检索
            use_vector: 是否使用 Qdrant 向量检索
            entity_filter: 实体过滤（仅用于图检索）
            metadata_filter: 元数据过滤（仅用于向量检索）
        
        Returns:
            {
                'graph_results': [...],      # 来自 Neo4j 的结果
                'vector_results': [...],     # 来自 Qdrant 的结果
                'merged_results': [...],     # 合并去重后的结果
                'time_distribution': {...}   # 时间分布统计
            }
        
        示例:
            # 查询最近7天内关于"GPT-4"的相关信息
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=7)
            
            results = pipeline.search_with_time_dimension(
                query="GPT-4的最新进展",
                start_time=start.isoformat(),
                end_time=end.isoformat(),
                top_k=10,
                entity_filter="GPT-4"
            )
        """
        from .neo4j_storage import Neo4jGraphStore
        
        graph_results = []
        vector_results = []
        
        # 1. Neo4j 图检索（基于时间范围）
        if use_graph:
            try:
                print(f"[时间维度检索] 使用 Neo4j 图检索（{start_time} ~ {end_time}）...")
                with Neo4jGraphStore(
                    uri=self.config.neo4j_uri,
                    user=self.config.neo4j_user,
                    password=self.config.neo4j_password
                ) as neo4j_store:
                    graph_results = neo4j_store.find_facts_by_timerange(
                        start_time=start_time,
                        end_time=end_time,
                        entity_filter=entity_filter,
                        limit=top_k
                    )
                    print(f"  - Neo4j 返回 {len(graph_results)} 条结果")
            except Exception as e:
                print(f"  ⚠️  Neo4j 检索失败: {e}")
        
        # 2. Qdrant 向量检索（基于时间范围 + 语义相似度）
        if use_vector:
            try:
                print(f"[时间维度检索] 使用 Qdrant 向量检索...")
                embedder = self.setup_embedder()
                
                with GauzRagVectorStore(
                    persist_directory=self.config.output_dir / "qdrant_db",
                    project_id=self.project_id,
                    use_server=(self.config.qdrant_mode == "server"),
                    server_url=self.config.qdrant_url
                ) as vector_store:
                    # 生成查询向量
                    query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
                    
                    # 向量检索
                    raw_results = vector_store.search_facts(
                        query_embedding,
                        top_k=top_k,
                        where=metadata_filter,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    # 格式化结果
                    for i, fid in enumerate(raw_results['ids']):
                        vector_results.append({
                            'fact_id': fid,
                            'content': raw_results['contents'][i],
                            'score': raw_results['scores'][i],
                            'created_at': raw_results['metadatas'][i].get('created_at', ''),
                            'source': 'vector',
                            'metadata': raw_results['metadatas'][i]
                        })
                    
                    print(f"  - Qdrant 返回 {len(vector_results)} 条结果")
            except Exception as e:
                print(f"  ⚠️  Qdrant 检索失败: {e}")
        
        # 3. 合并去重（以 fact_id 为准）
        fact_id_to_result = {}
        
        # 优先保留图检索结果（因为有实体关系信息）
        for result in graph_results:
            fid = result['fact_id']
            fact_id_to_result[fid] = {
                'fact_id': fid,
                'content': result['content'],
                'created_at': str(result.get('created_at', '')),
                'entities': result.get('entities', []),
                'communities': result.get('communities', []),
                'source': 'graph',
                'score': 0.0  # 图检索没有相似度分数
            }
        
        # 添加向量检索结果（如果不存在）或更新分数
        for result in vector_results:
            fid = result['fact_id']
            if fid in fact_id_to_result:
                # 已存在，补充向量分数
                fact_id_to_result[fid]['score'] = result['score']
                fact_id_to_result[fid]['source'] = 'graph+vector'
            else:
                # 新增
                fact_id_to_result[fid] = result
        
        # 按分数排序
        merged_results = sorted(
            fact_id_to_result.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # 4. 时间分布统计（可选）
        time_distribution = {}
        if use_graph:
            try:
                with Neo4jGraphStore(
                    uri=self.config.neo4j_uri,
                    user=self.config.neo4j_user,
                    password=self.config.neo4j_password
                ) as neo4j_store:
                    time_distribution = neo4j_store.get_time_distribution(granularity="day")
            except Exception as e:
                print(f"  ⚠️  时间分布统计失败: {e}")
        
        print(f"[时间维度检索] 完成！合并后共 {len(merged_results)} 条结果")
        
        return {
            'graph_results': graph_results,
            'vector_results': vector_results,
            'merged_results': merged_results,
            'time_distribution': time_distribution
        }
    
    # ===== 完整流程 =====
    
    
    # ===== 分步骤执行 =====
    
    
    def build_conversation_embedding_sync(
        self,
        conversation_id: int,
        conversation_text: str,
        project_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        同步构建 Conversation Embedding（中期记忆）
        
        Args:
            conversation_id: 对话ID
            conversation_text: 对话原文
            project_id: 项目ID
            metadata: 元数据
        """
        embedder = self.setup_embedder()
        
        with GauzRagVectorStore(
            persist_directory=self.config.output_dir / "qdrant_db",
            project_id=self.project_id,
            use_server=(self.config.qdrant_mode == "server"),
            server_url=self.config.qdrant_url
        ) as vector_store:
            # 生成 embedding
            print(f"    - 生成 Conversation Embedding...")
            conv_embedding = embedder.encode(
                [conversation_text],
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
            
            # 构建 metadata
            conv_metadata = {
                "content_type": "conversation",
                "source_identifier": f"conv_{conversation_id}",
                **(metadata or {})
            }
            
            # 写入 Qdrant（conversation 集合）
            vector_store.add_conversation(
                conversation_id=conversation_id,
                text=conversation_text,
                embedding=conv_embedding,
                metadata=conv_metadata
            )
            print(f"    - 已写入 Qdrant conversation 集合")
    
    def build_fact_embeddings(
        self,
        new_facts: List[Dict[str, Any]],
        conversation_id: int,
        conversation_text: str,
        skip_conversation_embedding: bool = False
    ) -> None:
        """
        构建/更新 Fact Embedding（基于 Qdrant，增量模式）
        
        Args:
            new_facts: 新提取的 facts（带 fact_id）
            conversation_id: 对话ID
            conversation_text: 对话原文
            skip_conversation_embedding: 是否跳过 conversation embedding（如果已在中期记忆阶段完成）
        """
        embedder = self.setup_embedder()
        
        # 使用 with 语句确保 VectorStore 正确关闭
        with GauzRagVectorStore(
            persist_directory=self.config.output_dir / "qdrant_db",
            project_id=self.project_id,
            use_server=(self.config.qdrant_mode == "server"),
            server_url=self.config.qdrant_url
        ) as vector_store:
            print(f"  - 新增 {len(new_facts)} 条 facts")
            print(f"  - 对话 ID: {conversation_id}")
            
            # 🚀 批量生成embeddings（Facts，可选Conversation）
            # 检查 conversation 是否需要embedding
            need_conv_embedding = (not skip_conversation_embedding) and (not vector_store.conversation_exists(conversation_id))
            
            # 收集所有需要embedding的文本
            all_texts = [f['content'] for f in new_facts]
            if need_conv_embedding:
                all_texts.append(conversation_text)  # 最后一个是conversation
                print(f"  - 注意: Conversation 未在中期记忆阶段完成，现在补充")
            
            print(f"  - 批量生成 embeddings: {len(new_facts)} Facts" + 
                  (f" + 1 Conversation" if need_conv_embedding else "") + "...")
            
            # 🚀 一次性批量调用Embedding API
            all_embeddings = embedder.encode(
                all_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # 分离embeddings
            fact_embeddings = all_embeddings[:len(new_facts)]
            conv_embedding = all_embeddings[-1] if need_conv_embedding else None
            
            # 添加到 Qdrant
            fact_ids = [f['fact_id'] for f in new_facts]
            conversation_ids = [conversation_id] * len(new_facts)
            fact_contents = [f['content'] for f in new_facts]
            
            # 获取 conversation 的 metadata
            db = self.setup_database()
            conn = db.get_connection()
            conv_metadata_dict = None  # 用于 conversation embedding
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT content_type, source_identifier, source_metadata
                        FROM conversations
                        WHERE conversation_id = %s
                    """, (conversation_id,))
                    conv_row = cur.fetchone()
                    
                    # 构建 metadata 列表（每个 fact 都附带 conversation 的 metadata）
                    fact_metadata = []
                    if conv_row:
                        import json
                        content_type, source_identifier, source_metadata_json = conv_row
                        source_metadata = json.loads(source_metadata_json) if source_metadata_json else {}
                        
                        # 构建完整的 metadata（用于 conversation）
                        conv_metadata_dict = {
                            "content_type": content_type,
                            "source_identifier": source_identifier,
                            **source_metadata  # 展开 metadata 所有字段
                        }
                        
                        # 为每个 fact 复制一份
                        for _ in new_facts:
                            fact_metadata.append(conv_metadata_dict.copy())
            finally:
                conn.close()
            
            vector_store.add_facts(
                fact_ids=fact_ids,
                contents=fact_contents,
                embeddings=fact_embeddings,
                conversation_ids=conversation_ids,
                metadata=fact_metadata if fact_metadata else None
            )
            print(f"  ✓ 已添加 {len(new_facts)} 条 facts 到 Qdrant（含 metadata）")
            
            # 添加 conversation embedding（如果需要）
            if need_conv_embedding and conv_embedding is not None:
                vector_store.add_conversation(
                    conversation_id=conversation_id,
                    text=conversation_text,
                    embedding=conv_embedding,
                    metadata=conv_metadata_dict
                )
                print(f"  ✓ 已添加 Conversation {conversation_id} 到 Qdrant（含 metadata）")
            else:
                print(f"  - Conversation {conversation_id} 已存在，跳过embedding")
            
            # 统计信息
            stats = vector_store.get_statistics()
            print(f"  - 当前统计:")
            print(f"    · Facts: {stats['facts_count']} 条")
            print(f"    · Conversations: {stats['conversations_count']} 条")
    
    async def build_topic_embeddings_from_neo4j(self, lightrag_builder) -> None:
        """
        构建 Topic Embedding（从 Neo4j 读取，支持 GauzRag）
        
        Args:
            lightrag_builder: LightRAGBuilder 实例（用于访问 Neo4j）
        """
        embedder = self.setup_embedder()
        
        # 使用 with 语句确保 VectorStore 正确关闭
        with GauzRagVectorStore(
            persist_directory=self.config.output_dir / "qdrant_db",
            project_id=self.project_id,
            use_server=(self.config.qdrant_mode == "server"),
            server_url=self.config.qdrant_url
        ) as vector_store:
            # 从 Neo4j 读取 Topics
            print(f"  - 从 Neo4j 读取 Topics...")
            topics = await lightrag_builder.neo4j_store.get_all_topics()
            
            if not topics:
                print(f"  ⚠️  Neo4j 中未找到 Topics，跳过 Topic embeddings")
                return
            
            print(f"  - 找到 {len(topics)} 个 Topics")
            
            # 生成 embeddings
            print(f"  - 生成 {len(topics)} 个 Topic embeddings...")
            topic_texts = [f"{t['title']}\n\n{t['summary']}" for t in topics]
            topic_embeddings = embedder.encode(
                topic_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # 添加到 Qdrant
            for i, topic in enumerate(topics):
                vector_store.add_community(
                    community_id=topic['topic_id'],
                    title=topic['title'],
                    summary=topic['summary'],
                    embedding=topic_embeddings[i]
                )
            
            print(f"  ✓ 已添加 {len(topics)} 个 Topics 到 Qdrant（数据源: Neo4j）")
            
            # 统计信息
            stats = vector_store.get_statistics()
            print(f"  - 当前统计:")
            print(f"    · Topics (Communities): {stats['communities_count']} 个")
    
    def build_topic_embeddings(self) -> None:
        """
        构建 Topic Embedding（旧版本，基于 JSON 文件）
        
        已弃用：请使用 build_topic_embeddings_from_neo4j()
        此方法仅用于向后兼容 GraphRAG 模式
        """
        import json
        
        embedder = self.setup_embedder()
        
        # 使用 with 语句确保 VectorStore 正确关闭
        with GauzRagVectorStore(
            persist_directory=self.config.output_dir / "qdrant_db",
            project_id=self.project_id,
            use_server=(self.config.qdrant_mode == "server"),
            server_url=self.config.qdrant_url
        ) as vector_store:
            # 读取 topic reports
            reports_path = self.config.output_dir / "fact_community_reports.json"
            if not reports_path.exists():
                print(f"  ⚠️  未找到 fact_community_reports.json，跳过 Topic embeddings")
                return
            
            with open(reports_path, 'r', encoding='utf-8') as f:
                reports = json.load(f)
            
            # 准备topic数据
            topics = []
            if isinstance(reports, dict):
                for key, report_data in reports.items():
                    comm_id = report_data.get('community_id', key)
                    if isinstance(comm_id, str) and comm_id.startswith('fact_community_'):
                        topic_id = int(comm_id.replace('fact_community_', ''))
                    else:
                        topic_id = comm_id
                    
                    report = report_data.get('report', {})
                    title = report.get('title', f'Topic {topic_id}')
                    summary = report.get('summary', '')
                    
                    # 组合title和summary作为embedding文本
                    topic_text = f"{title}\n\n{summary}"
                    topics.append({
                        'topic_id': topic_id,
                        'title': title,
                        'summary': summary,
                        'text': topic_text
                    })
            
            if not topics:
                print(f"  ⚠️  未找到 Topics，跳过 Topic embeddings")
                return
            
            print(f"  - 生成 {len(topics)} 个 Topic embeddings...")
            topic_texts = [t['text'] for t in topics]
            topic_embeddings = embedder.encode(
                topic_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # 添加到 Qdrant
            for i, topic in enumerate(topics):
                vector_store.add_community(
                    community_id=topic['topic_id'],
                    title=topic['title'],
                    summary=topic['summary'],
                    embedding=topic_embeddings[i]
                )
            
            print(f"  ✓ 已添加 {len(topics)} 个 Topics 到 Qdrant（数据源: JSON - Legacy）")
            
            # 统计信息
            stats = vector_store.get_statistics()
            print(f"  - 当前统计:")
            print(f"    · Topics (Communities): {stats['communities_count']} 个")
    
    def refine_bundle_with_llm(
        self,
        query: str,
        bundle: Any  # BundleResponse类型
    ) -> Dict[str, str]:
        """
        使用LLM精炼Bundle内容
        
        Args:
            query: 用户查询
            bundle: Bundle对象（包含conversations, facts, topics）
        
        Returns:
            {"related_memory": "...", "quote": "..."}
        """
        import json
        import requests
        
        print(f"    · 精炼 Bundle {bundle.bundle_id}...")
        
        # Construct Topics section
        topics_text = ""
        if bundle.topics:
            topics_list = [f"- {t.title}: {t.summary}" for t in bundle.topics]
            topics_text = "\n".join(topics_list)
        else:
            topics_text = "(No topic information)"
        
        # Construct Facts section (simplified: only keep content, hop_facts' content+relation, timestamp)
        facts_text = ""
        if bundle.facts:
            facts_list = []
            for f in bundle.facts:
                # Main Fact content
                fact_line = f"- {f.content}"
                
                # Add timestamp if available
                if f.metadata and f.metadata.get('timestamp'):
                    fact_line += f" [{f.metadata['timestamp']}]"
                
                # Add multi-hop extensions (only keep content and relation)
                if f.hop_facts and isinstance(f.hop_facts, dict):
                    context_lines = []
                    for hop_key in sorted(f.hop_facts.keys()):
                        hop_facts = f.hop_facts[hop_key]
                        for hf in hop_facts:
                            # Only keep content and relation
                            relation = hf.get('relation', 'RELATED')
                            content = hf.get('content', '')
                            if content:
                                context_lines.append(f"    [{relation}] {content}")
                    
                    if context_lines:
                        fact_line += "\n" + "\n".join(context_lines)
                
                facts_list.append(fact_line)
            
            facts_text = "\n".join(facts_list)
        else:
            facts_text = "(No fact information)"
        
        # Construct Dialogues section (simplified: keep text and timestamp)
        dialogues_text = ""
        if bundle.conversations:
            dialogue_snippets = []
            for i, conv in enumerate(bundle.conversations[:3]):  # Max 3 dialogues
                # Truncate to first 300 characters
                snippet = conv.text[:300] + "..." if len(conv.text) > 300 else conv.text
                
                # Add timestamp
                if conv.metadata and conv.metadata.get('timestamp'):
                    snippet = f"[{conv.metadata['timestamp']}] {snippet}"
                
                dialogue_snippets.append(snippet)
            dialogues_text = "\n\n".join(dialogue_snippets)
        else:
            dialogues_text = "(No dialogue information)"
        
        # Construct complete prompt
        system_prompt = """You are a Query-Focused Memory Distiller. Your goal is to extract "Atomic Intelligence" from the provided context that specifically answers OR contributes evidence to the User's Core Question.

### INPUT DATA STRUCTURE:
- <topics>: High-level summaries of the conversation.
- <facts>: Structured facts extracted from the conversation, including timestamps.
- <dialogues>: Raw conversation snippets with timestamps.

### STRICT REFINEMENT RULES:
1. **Relevance Filter (INCLUSIVE, CONTEXTUAL & INFERENTIAL)**:
   - **Core Principle**: If the bundle contains *any* evidence, partial clues, or context that helps construct an answer (even if it contradicts the premise of the question), **EXTRACT IT**.
   - **CRITICAL: Competing & Alternative Evidence**: 
     - If the query specifically asks about "Target A" (e.g., "Does the user play tennis?"), but the text contains strong evidence for an exclusive "Target B" (e.g., "User explicitly states they only swim and hate racket sports"), **YOU MUST EXTRACT TARGET B**.
     - Treat these "competing facts" as highly relevant because they provide the actual state of affairs and help answer "No" with evidence.
   - **Partial Matches**: If the query asks for a specific attribute but the text only confirms the general event occurred, retain it.
   - **Irrelevance**: Only return empty strings if the bundle is completely unrelated to the *domain* of the query (e.g., discussing "Cooking recipes" when the query is about "Political views").

2. **Denoising & Consolidation**: 
   - Extract specific details (dates, names, locations, motives) required by the query.
   - Discard purely social pleasantries ("Hi", "How are you") unless they contain embedded facts.
   - Merge multiple facts into a single, dense, coherent paragraph.
   - **Contextualization**: If extracting competing evidence, phrase it to show contrast (e.g., "While no mention of [Query Topic], the context confirms [Actual Topic]").

3. **Temporal Resolution**: 
   - Use the provided timestamps (e.g., `[2023-04-18]`) to convert relative terms like "yesterday" or "last month" into absolute dates/months within the `related_memory`.

4. **Quote Extraction**: 
   - For the `quote` field, locate the most distinct raw text snippet (preferably from <dialogues>) that provides evidence or context. Keep the timestamp in the quote if helpful.

### OUTPUT FORMAT:
Return a single valid JSON object:
{
    "related_memory": "The refined atomic intelligence paragraph. Include competing/alternative facts if direct matches are missing. (Empty string if irrelevant)",
    "quote": "Verbatim evidence string. (Empty string if irrelevant)"
}
"""
        
        user_prompt = f"""User's Core Question: "{query}"

--- Memory Bundle ---

[Topics]:
{topics_text}

[Key Facts]:
{facts_text}

[Dialogues]:
{dialogues_text}

Please analyze and refine."""
        
        # 调用LLM
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                f"{self.config.llm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析JSON
            # 尝试提取JSON（可能被markdown包裹）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            refined_data = json.loads(content)
            
            # 确保返回格式正确
            result = {
                "related_memory": refined_data.get("related_memory", ""),
                "quote": refined_data.get("quote")
            }
            print(f"    ✓ Bundle {bundle.bundle_id} 完成")
            return result
        
        except Exception as e:
            print(f"    ✗ Bundle {bundle.bundle_id} 失败: {str(e)}")
            # 返回降级结果
            return {
                "related_memory": "（LLM精炼失败，请查看原始数据）",
                "quote": None
            }
    
    def refine_recent_turns_with_llm(
        self,
        query: str,
        recent_turns: Any  # RecentTurnsBundle类型
    ) -> str:
        """
        使用LLM精炼最近对话轮次（多轮对话上下文）
        
        Args:
            query: 用户当前查询
            recent_turns: 最近对话Bundle（包含最近N轮的conversations）
        
        Returns:
            精炼后的对话上下文总结
        """
        import json
        import requests
        
        print(f"    · 精炼最近 {len(recent_turns.conversations)} 轮对话...")
        
        # Construct dialogue history
        dialogues_text = ""
        if recent_turns.conversations:
            dialogue_list = []
            for conv in recent_turns.conversations:
                # Extract turn information
                turn = conv.metadata.get('turn', '?') if conv.metadata else '?'
                dialogue_list.append(f"[Turn {turn}] {conv.text}")
            dialogues_text = "\n".join(dialogue_list)
        else:
            dialogues_text = "(No dialogue history)"
        
        # Construct prompt
        system_prompt = """You are a dialogue context analysis expert preparing context for factoid question answering. Your tasks:
Analyze recent dialogue turns and extract contextual information and historical background relevant to the current question.

Summarize in one paragraph:
1. What topics did the user discuss previously?
2. Which historical information is directly relevant to the current question?
3. Is the current question a continuation or follow-up of previous conversations?

⚠️ CRITICAL OUTPUT REQUIREMENTS (for LoCoMo QA Benchmark):
- Write in a FACT-DENSE, PHRASE-ORIENTED style
- Prioritize KEY ENTITIES, DATES, and FACTS
- For dates: Use ENGLISH format ONLY (e.g., "7 May 2023", "June 2023"), NEVER Chinese like "2023年5月7日"
- Extract absolute dates from time annotations in parentheses (e.g., "yesterday (May 7, 2023)" → use "May 7, 2023")
- Minimize redundant words

If the dialogue history is completely irrelevant to the current question, return an empty string. Always write in ENGLISH."""
        
        user_prompt = f"""Current Question: "{query}"

Recent Dialogue History:
{dialogues_text}

Please analyze and summarize the relevant dialogue context."""
        
        # 调用LLM
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        try:
            response = requests.post(
                f"{self.config.llm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            print(f"    ✓ recent_turns 精炼完成")
            return content
        
        except Exception as e:
            print(f"    ✗ recent_turns 精炼失败: {str(e)}")
            return "（对话上下文精炼失败）"
    
    def refine_short_term_with_llm(
        self,
        query: str,
        short_term: Any  # ShortTermMemory类型
    ) -> str:
        """
        使用LLM精炼短期记忆（indexed=0的对话）
        
        Args:
            query: 用户当前查询
            short_term: 短期记忆（包含未完成索引的conversations）
        
        Returns:
            精炼后的短期记忆摘要
        """
        import json
        import requests
        
        print(f"    · 精炼短期记忆（{len(short_term.conversations)} 条对话）...")
        
        # 构建对话文本
        dialogues_text = ""
        if short_term.conversations:
            dialogue_list = []
            for i, conv in enumerate(short_term.conversations, 1):
                timestamp = conv.get('metadata', {}).get('timestamp', '') if conv.get('metadata') else ''
                time_str = f" [{timestamp}]" if timestamp else ""
                dialogue_list.append(f"[{i}]{time_str} {conv['text']}")
            dialogues_text = "\n\n".join(dialogue_list)
        else:
            dialogues_text = "(No short-term memory)"
        
        # 构建提示
        system_prompt = """You are a Query-Focused Memory Distiller for short-term memory (recent conversations not yet fully indexed).

Your task: Extract and consolidate information from recent conversations that is DIRECTLY relevant to answering the user's query.

### REFINEMENT RULES:
1. **Query-Centric**: Only extract facts, dates, entities, or context that help answer the query
2. **Denoise**: Remove greetings, filler words, and irrelevant chitchat
3. **Consolidate**: If multiple conversations mention the same thing, merge into one statement
4. **Preserve Details**: Keep specific names, dates, numbers, and key facts
5. **Format**: Write as a dense, factual paragraph (not bullet points)

### DATE HANDLING:
- Use ENGLISH date format ONLY (e.g., "7 May 2023", "June 2023")
- Extract absolute dates from parentheses: "yesterday (May 7, 2023)" → "May 7, 2023"
- NEVER use Chinese date formats

### OUTPUT:
Return ONLY the refined memory paragraph. If nothing is relevant, return an empty string.
Always write in ENGLISH."""
        
        user_prompt = f"""User's Query: "{query}"

Recent Conversations (Short-term Memory):
{dialogues_text}

Please extract and consolidate relevant information."""
        
        # 调用LLM
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 400
        }
        
        try:
            response = requests.post(
                f"{self.config.llm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            print(f"    ✓ short_term 精炼完成")
            return content
        
        except Exception as e:
            print(f"    ✗ short_term 精炼失败: {str(e)}")
            return "（短期记忆精炼失败）"
    
    def mark_conversation_indexed(self, conversation_id: int) -> None:
        """
        标记对话为已索引
        
        Args:
            conversation_id: 对话 ID
        """
        db = self.setup_database()
        db.update_conversation_indexed(conversation_id, indexed=True)
        print(f"对话 {conversation_id} 已标记为已索引")
    
    def build_entity_community_mapping(self) -> Dict[str, List[Dict]]:
        """构建实体-社区映射"""
        mapper = self.setup_entity_community_mapper()
        
        # 构建映射
        entity_to_communities = mapper.build_mapping()
        
        # 保存映射
        output_path = self.config.output_dir / "entity_to_communities.json"
        mapper.save_mapping(output_path)
        
        return entity_to_communities
    
    def detect_topics_with_semantic_clustering(
        self,
        similarity_threshold: float = 0.75,
        min_topic_size: int = 3,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        使用方案3（向量聚类）检测 Topics
        
        这是替代 lightrag_builder.detect_fact_topics() 的新方法
        
        优势：
        - 基于向量相似度，避免连通分量的"语义漂移"
        - LLM 总结质量最高
        - 增量友好
        
        Args:
            similarity_threshold: 相似度阈值（默认 0.75）
            min_topic_size: 最小 Topic 大小（默认 3）
            force_rebuild: 是否强制重建（删除旧 Topics）
        
        Returns:
            {'total_facts': int, 'total_topics': int, 'topics': [...]}
        """
        from .semantic_topic_detector import SemanticTopicDetector
        from openai import OpenAI
        import os
        
        print(f"\n{'='*60}")
        print(f"方案3：向量聚类 Topic 检测")
        print(f"{'='*60}")
        
        # 初始化检测器
        detector = SemanticTopicDetector(
            db_manager=self.setup_database(),
            vector_store=self.setup_vector_store(),
            embedder=self.setup_embedder(),
            llm_client=OpenAI(
                api_key=os.getenv("GAUZ_LLM_API_KEY"),
                base_url=os.getenv("GAUZ_LLM_API_BASE")
            ),
            llm_model=os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini"),
            similarity_threshold=similarity_threshold,
            min_topic_size=min_topic_size
        )
        
        # 批量聚类所有 Facts
        result = detector.batch_cluster_all_facts(
            project_id=self.project_id,
            force_rebuild=force_rebuild
        )
        
        print(f"\n✓ Topic 检测完成：{result['total_facts']} Facts → {result['total_topics']} Topics")
        
        return result
    
    def extract_fact_entities(self) -> Dict[int, List[str]]:
        """提取所有 Facts 的实体"""
        extractor = self.setup_fact_entity_extractor()
        
        # 提取
        fact_to_entities = extractor.extract_all()
        
        # 保存
        output_path = self.config.output_dir / "fact_to_entities.json"
        extractor.save(output_path)
        
        return fact_to_entities
    
    def build_fact_relations_for_new_fact(
        self,
        new_fact: Dict[str, Any],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        为新 Fact 构建与现有 Facts 的关系
        
        Args:
            new_fact: 新 Fact，需包含 fact_id, content, entities
            threshold: 社区置信度阈值
        
        Returns:
            关系列表
        """
        # 1. 加载或构建实体-社区映射
        mapping_path = self.config.output_dir / "entity_to_communities.json"
        if mapping_path.exists():
            mapper = self.setup_entity_community_mapper()
            entity_to_communities = mapper.load_mapping(mapping_path)
        else:
            print("实体-社区映射不存在，正在构建...")
            entity_to_communities = self.build_entity_community_mapping()
        
        # 2. 定位新 Fact 关联的社区
        locator = FactCommunityLocator(entity_to_communities)
        related_communities = locator.locate(new_fact.get('entities', []), threshold)
        
        if not related_communities:
            print(f"⚠️  Fact {new_fact['fact_id']} 没有关联到任何社区")
            return []
        
        print(f"\nFact {new_fact['fact_id']} 关联到 {len(related_communities)} 个社区:")
        for comm in related_communities:
            print(f"  - Community {comm['community_id']}: "
                  f"置信度 {comm['confidence']:.2f}, "
                  f"实体 {comm['entities']}")
        
        # 3. 收集候选 Facts
        candidate_facts = self._collect_candidate_facts(new_fact, related_communities)
        
        if not candidate_facts:
            print(f"⚠️  没有找到候选 Facts")
            return []
        
        print(f"收集到 {len(candidate_facts)} 条候选 Facts")
        
        # 4. 用 LLM 分析关系
        analyzer = self.setup_fact_relation_analyzer()
        relations = analyzer.analyze_batch(new_fact, candidate_facts)
        
        return relations
    
    def _collect_candidate_facts(
        self,
        new_fact: Dict[str, Any],
        related_communities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """收集候选 Facts"""
        # 加载 community_facts.json
        community_facts_path = self.config.output_dir / "community_facts.json"
        if not community_facts_path.exists():
            return []
        
        import json
        with open(community_facts_path, 'r', encoding='utf-8') as f:
            community_facts_data = json.load(f)
        
        candidate_facts = {}
        new_fact_id = new_fact['fact_id']
        new_fact_entities = set(new_fact.get('entities', []))
        
        for comm_info in related_communities:
            comm_id = comm_info['community_id']
            comm_key = f"community_{comm_id}"
            
            if comm_key not in community_facts_data:
                continue
            
            facts = community_facts_data[comm_key]['facts']
            
            for fact in facts:
                fact_id = fact['fact_id']
                
                # 跳过自己
                if fact_id == new_fact_id:
                    continue
                
                if fact_id not in candidate_facts:
                    candidate_facts[fact_id] = {
                        'fact_id': fact_id,
                        'content': fact['content'],
                        'communities': [],
                        'shared_entities': []
                    }
                
                candidate_facts[fact_id]['communities'].append({
                    'community_id': comm_id,
                    'confidence': comm_info['confidence']
                })
                
                # 计算共享实体（从社区的实体列表）
                shared = set(comm_info['entities']) & new_fact_entities
                candidate_facts[fact_id]['shared_entities'].extend(list(shared))
        
        # 去重共享实体
        for fact_id in candidate_facts:
            candidate_facts[fact_id]['shared_entities'] = list(
                set(candidate_facts[fact_id]['shared_entities'])
            )
        
        return list(candidate_facts.values())
    
    def add_fact_to_relation_graph(
        self,
        new_fact: Dict[str, Any],
        relations: List[Dict[str, Any]]
    ) -> None:
        """将新 Fact 和关系添加到图中"""
        graph_builder = self.setup_fact_relation_graph()
        
        # 添加节点
        graph_builder.add_fact_node(new_fact)
        
        # 添加边
        graph_builder.add_relations(relations)
        
        # 保存
        graph_path = self.config.output_dir / "fact_relations.json"
        graph_builder.save_graph(graph_path)
    
    async def build_fact_relations_with_lightrag(
        self,
        new_facts: List[Dict[str, Any]],
        lightrag_builder: 'GauzRagBuilder',
        fact_entities_map: Optional[Dict[int, List[Dict]]] = None,
        skip_same_batch: bool = True  # 新增参数：跳过同批次分析
    ) -> None:
        """
        使用 GauzRag 实体映射快速构建 Facts 语义关系
        
        流程：
        1. 通过实体图快速找到候选 Facts（O(k)）
        2. 使用 LLM 分析语义关系类型（因果、时序、支持等）
        3. 存储到 Neo4j
        
        Args:
            new_facts: 新 Facts 列表（需包含 fact_id, content）
            lightrag_builder: GauzRag 构建器实例
            fact_entities_map: 实体映射（可选）
            skip_same_batch: 是否跳过同批次分析（默认True，因为已通过显性关系提取处理）
        """
        if not GAUZRAG_AVAILABLE or not GauzRagEntityMapper:
            print("  ⚠️  GauzRag 不可用，无法使用实体映射")
            return
        
        print(f"\n[4/4] 构建 Fact 语义关系（实体图1跳 + LLM 分析）...")
        
        if skip_same_batch:
            print(f"  ℹ️  跳过同批次分析（已通过显性关系提取完成）")
        
        try:
            # 步骤1: 为每个新 Fact 使用 LLM 分析语义关系
            from .fact_relation_builder import FactRelationAnalyzer
            
            analyzer = FactRelationAnalyzer(
                api_base=self.config.llm_api_base,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                temperature=0.3
            )
            
            total_relations = 0
            relation_types_count = {}
            
            # 收集本批次fact_ids（用于过滤）
            batch_fact_ids = {fact['fact_id'] for fact in new_facts}
            
            # 并行处理所有Facts的关系分析
            async def process_single_fact(fact_idx: int, new_fact: Dict):
                """处理单个Fact的关系分析"""
                fact_id = new_fact['fact_id']
                
                # 策略1: 内存注入 - 同批次Facts（根据 skip_same_batch 决定是否启用）
                same_batch_candidates = []
                if not skip_same_batch and fact_entities_map:
                    # 只有在 skip_same_batch=False 时才分析同批次
                    for other_fact in new_facts:
                        if other_fact['fact_id'] == fact_id:
                            continue  # 跳过自己
                        
                        other_entities = [e['name'] for e in fact_entities_map.get(other_fact['fact_id'], [])]
                        current_entities = [e['name'] for e in fact_entities_map.get(fact_id, [])]
                        shared = list(set(current_entities) & set(other_entities))
                        
                        same_batch_candidates.append({
                            'fact_id': other_fact['fact_id'],
                            'content': other_fact['content'],
                            'entities': other_entities,
                            'shared_entities': shared
                        })
                
                # 策略2: 实体1跳 - 历史Facts（只查询历史数据）
                entity_1hop_candidates = await lightrag_builder.neo4j_store.get_candidate_facts(
                    fact_id,
                    max_candidates=15,  # 限制候选数量
                    use_community_optimization=False
                )
                
                # 过滤重复（排除同批次）
                entity_1hop_candidates = [
                    c for c in entity_1hop_candidates 
                    if c['fact_id'] not in batch_fact_ids
                ]
                
                # 合并候选
                all_candidates = same_batch_candidates + entity_1hop_candidates
                
                if not all_candidates:
                    return []
                
                batch_info = f"同批次={len(same_batch_candidates)}, " if same_batch_candidates else ""
                print(f"  [{fact_idx}/{len(new_facts)}] Fact {fact_id}: {batch_info}历史1跳={len(entity_1hop_candidates)}")
                
                # 准备Fact信息（包含实体）
                fact_with_entities = {
                    **new_fact,
                    'entities': [e['name'] for e in fact_entities_map.get(fact_id, [])] if fact_entities_map else []
                }
                
                # 使用 LLM 分析语义关系
                import asyncio
                relations = await asyncio.to_thread(
                    analyzer.analyze_batch, 
                    fact_with_entities, 
                    all_candidates, 
                    batch_size=30
                )
                
                if relations:
                    print(f"       ✓ Fact {fact_id} 发现 {len(relations)} 条语义关系")
                
                return relations
            
            # 并行执行所有Facts的分析
            print(f"  🚀 并行分析 {len(new_facts)} 个Facts的关系...")
            results = await asyncio.gather(*[
                process_single_fact(i+1, fact) 
                for i, fact in enumerate(new_facts)
            ])
            
            # 3. 批量存储所有关系到 Neo4j
            for relations in results:
                for rel in relations:
                    await lightrag_builder.neo4j_store.add_semantic_relation(rel)
                    
                    # 统计关系类型
                    rel_type = rel.get('relation_type', 'unknown')
                    relation_types_count[rel_type] = relation_types_count.get(rel_type, 0) + 1
                
                total_relations += len(relations)
            
            print(f"\n  ✓ Fact 语义关系构建完成！")
            print(f"    - LLM分析的语义关系数: {total_relations}")
            
            if relation_types_count:
                print(f"    - 关系类型分布:")
                for rel_type, count in sorted(relation_types_count.items(), key=lambda x: x[1], reverse=True):
                    print(f"      · {rel_type}: {count} 条")
            
            # 查询完整的图谱统计（包括显性关系）
            stats = await lightrag_builder.neo4j_store.get_fact_graph_stats()
            print(f"\n  📊 Neo4j图谱统计（包括显性关系）:")
            print(f"    - Fact节点数: {stats['nodes']}")
            print(f"    - 总关系数: {stats['edges']}")
            
            # 导出到 JSON（用于 Community Detection 等遗留功能）
            if stats['nodes'] >= 5:
                print(f"\n  - 导出到 JSON（用于 Community Detection）...")
                json_path = self.config.output_dir / "fact_relations.json"
                await lightrag_builder.neo4j_store.export_semantic_relations_to_json(json_path)
                print(f"    ✓ 已导出到: {json_path}")
            
        except Exception as e:
            print(f"  ⚠️  构建 Fact Relations 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== 查询召回 =====
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        查询召回（Community 级别）
        
        Args:
            query: 用户查询
            top_k: 返回前 k 个社区
        
        Returns:
            搜索结果列表（返回所有 facts，不去重）
        """
        searcher = self.setup_searcher()
        
        top_k = top_k or self.config.search_top_k
        
        return searcher.search(query=query, top_k=top_k)
    
    def search_facts(
        self,
        query: str,
        top_k: int = 10,
        include_community: bool = True,
        search_mode: str = "hybrid",
        metadata_filter: dict = None
    ) -> List[Dict[str, Any]]:
        """
        查询召回（Fact 级别 - 基于 Qdrant）
        
        Args:
            query: 用户查询
            top_k: 返回前 k 个 facts
            include_community: 是否包含所属 community 的 report（暂未实现）
            search_mode: 搜索模式 - "fact" (Fact内容), "conversation" (对话原文), "hybrid" (混合，默认)
            metadata_filter: 元数据过滤条件
        
        Returns:
            搜索结果列表：
            [
                {
                    'fact_id': int,
                    'content': str,
                    'relevance_score': float,
                    'conversation_id': int
                },
                ...
            ]
        """
        # 使用新的 Qdrant 方法
        results = self.search_facts_chromadb(
            query=query,
            top_k=top_k,
            search_mode=search_mode,
            metadata_filter=metadata_filter
        )
        
        # 格式化为 API 预期的格式
        formatted_results = []
        for result in results:
            formatted_results.append({
                'fact_id': result['fact_id'],
                'content': result['content'],
                'relevance_score': result['score'],
                'conversation_id': result.get('conversation_id'),
                'source': result.get('source', 'qdrant'),
                'metadata': result.get('metadata')  # ← 添加 metadata
            })
        
        # TODO: 如果 include_community=True，添加 community 信息
        # 需要从 community_facts.json 或数据库读取
        
        return formatted_results

