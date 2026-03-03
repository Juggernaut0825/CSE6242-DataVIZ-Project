"""
向量存储模块 - 基于 Qdrant
专业向量数据库，支持大规模数据
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime


class GauzRagVectorStore:
    """GauzRag 向量存储管理器（基于 Qdrant）"""
    
    def __init__(
        self, 
        persist_directory: Path, 
        project_id: str, 
        vector_dim: int = 1024,
        use_server: bool = False,
        server_url: str = "http://localhost:6333"
    ):
        """
        初始化向量存储
        
        Args:
            persist_directory: 持久化目录（本地模式使用）
            project_id: 项目ID（用于隔离不同项目）
            vector_dim: 向量维度（默认 1024）
            use_server: 是否使用 Server 模式（支持并发）
            server_url: Server 地址（use_server=True 时使用）
        """
        self.project_id = project_id
        self.persist_directory = persist_directory
        self.vector_dim = vector_dim
        self.use_server = use_server
        self.server_url = server_url
        
        # 初始化 Qdrant 客户端
        if use_server:
            # Server 模式：支持并发读写
            print(f"  → 使用 Qdrant Server 模式: {server_url}")
            self.client = QdrantClient(url=server_url)
        else:
            # 本地模式：简单但不支持并发
            print(f"  → 使用 Qdrant 本地模式: {persist_directory}")
            self.client = QdrantClient(path=str(persist_directory))
        
        # 创建/获取 Collections（使用余弦相似度）
        self._ensure_collections_exist()
    
    def _ensure_collections_exist(self):
        """确保所有 Collection 存在"""
        collections = {
            f"{self.project_id}_facts": "Fact embeddings",
            f"{self.project_id}_conversations": "Conversation embeddings",
            f"{self.project_id}_communities": "Community embeddings"
        }
        
        existing = {col.name for col in self.client.get_collections().collections}
        
        for name, description in collections.items():
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    )
                )
    
    def close(self):
        """关闭 Qdrant 客户端，释放文件锁"""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except Exception as e:
                print(f"Warning: Failed to close Qdrant client: {e}")
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动关闭客户端"""
        self.close()
    
    def __del__(self):
        """析构时关闭客户端"""
        self.close()
    
    # ===== Fact 操作 =====
    
    def add_facts(
        self,
        fact_ids: List[int],
        contents: List[str],
        embeddings: np.ndarray,
        conversation_ids: List[int],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 100
    ) -> None:
        """
        批量添加 Facts（分批写入，避免超时）
        
        Args:
            fact_ids: Fact ID 列表
            contents: Fact 内容列表
            embeddings: Embedding 矩阵 (N, 1024)
            conversation_ids: 对话ID列表
            metadata: 额外的元数据
            batch_size: 每批写入的数量（默认100）
        """
        total_facts = len(fact_ids)
        current_timestamp = datetime.now().timestamp()
        
        # 分批处理
        for batch_start in range(0, total_facts, batch_size):
            batch_end = min(batch_start + batch_size, total_facts)
            points = []
            
            for i in range(batch_start, batch_end):
                fid = fact_ids[i]
                content = contents[i]
                conv_id = conversation_ids[i]
                
                payload = {
                    "fact_id": fid,
                    "content": content,
                    "conversation_id": conv_id,
                    "project_id": self.project_id,
                    "created_at": current_timestamp,
                    "created_at_iso": datetime.now().isoformat()
                }
                if metadata and i < len(metadata):
                    payload.update(metadata[i])
                
                points.append(PointStruct(
                    id=fid,
                    vector=embeddings[i].tolist(),
                    payload=payload
                ))
            
            # 批量写入当前batch
            self.client.upsert(
                collection_name=f"{self.project_id}_facts",
                points=points
            )
            
            if batch_end < total_facts:
                print(f"    · 已写入 {batch_end}/{total_facts} 条facts到Qdrant...")
        
        if total_facts > batch_size:
            print(f"    ✓ 全部 {total_facts} 条facts已写入Qdrant")
    
    def search_facts(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: Optional[Dict] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        搜索相似 Facts（支持时间维度过滤）
        
        Args:
            query_embedding: 查询向量 (1024,)
            top_k: 返回数量
            where: 元数据过滤条件，如 {'username': 'user_123', 'content_type': 'conversation'}
            start_time: 开始时间（ISO 格式），如 "2024-01-01T00:00:00"
            end_time: 结束时间（ISO 格式），如 "2024-12-31T23:59:59"
        
        Returns:
            {
                'ids': [...],
                'scores': [...],
                'metadatas': [...],
                'contents': [...]
            }
        
        示例:
            # 普通向量检索
            store.search_facts(query_emb, top_k=10)
            
            # 带时间范围的向量检索
            store.search_facts(
                query_emb, 
                top_k=10,
                start_time="2024-01-01T00:00:00",
                end_time="2024-12-31T23:59:59"
            )
        """
        from qdrant_client.models import Range
        
        # 构建 Qdrant 过滤器
        conditions = []
        
        # 添加 where 条件
        if where:
            for key, value in where.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        # 添加时间范围过滤（转换为 Unix 时间戳）
        # 注意：Qdrant 的 Range 过滤器只支持数字类型
        if start_time:
            try:
                # 将 ISO 格式转换为 Unix 时间戳（数字）
                start_timestamp = datetime.fromisoformat(start_time).timestamp()
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=Range(gte=start_timestamp)
                    )
                )
                print(f"    → 时间过滤 (start): {start_time} → timestamp: {start_timestamp}")
            except Exception as e:
                print(f"  ⚠️  start_time 格式错误: {e}")
        
        if end_time:
            try:
                # 将 ISO 格式转换为 Unix 时间戳（数字）
                end_timestamp = datetime.fromisoformat(end_time).timestamp()
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=Range(lte=end_timestamp)
                    )
                )
                print(f"    → 时间过滤 (end): {end_time} → timestamp: {end_timestamp}")
            except Exception as e:
                print(f"  ⚠️  end_time 格式错误: {e}")
        
        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=f"{self.project_id}_facts",
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        ).points
        
        # 转换格式
        return {
            'ids': [hit.id for hit in results],
            'scores': [hit.score for hit in results],  # Qdrant 返回的是相似度分数（0-1，越大越相似）
            'metadatas': [hit.payload for hit in results],
            'contents': [hit.payload.get('content', '') for hit in results]
        }
    
    def get_facts_count(self) -> int:
        """获取 Facts 总数"""
        collection_info = self.client.get_collection(f"{self.project_id}_facts")
        return collection_info.points_count
    
    def delete_facts(self, fact_ids: List[int]) -> None:
        """删除指定 Facts"""
        self.client.delete(
            collection_name=f"{self.project_id}_facts",
            points_selector=fact_ids
        )
    
    def get_facts_by_timerange(
        self,
        start_time: str,
        end_time: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        按时间范围获取 Facts（不使用向量检索，纯时间过滤）
        
        Args:
            start_time: 开始时间（ISO 格式）
            end_time: 结束时间（ISO 格式）
            limit: 返回数量限制
            offset: 偏移量（用于分页）
        
        Returns:
            Facts 列表
        
        示例:
            # 获取最近7天的所有 facts
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=7)
            facts = store.get_facts_by_timerange(
                start.isoformat(),
                end.isoformat()
            )
        """
        from qdrant_client.models import Range
        
        # 将 ISO 时间转换为 Unix 时间戳（数字）
        try:
            start_timestamp = datetime.fromisoformat(start_time).timestamp()
            end_timestamp = datetime.fromisoformat(end_time).timestamp()
        except Exception as e:
            print(f"  ⚠️  时间格式错误: {e}")
            return []
        
        # 构建时间过滤器（使用数字范围）
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="created_at",
                    range=Range(gte=start_timestamp, lte=end_timestamp)
                )
            ]
        )
        
        # 使用 scroll 获取数据（不需要向量检索）
        results, _ = self.client.scroll(
            collection_name=f"{self.project_id}_facts",
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False  # 不需要返回向量
        )
        
        return [
            {
                'fact_id': point.id,
                'content': point.payload.get('content', ''),
                'created_at': point.payload.get('created_at', ''),
                'metadata': point.payload
            }
            for point in results
        ]
    
    # ===== Conversation 操作 =====
    
    def add_conversation(
        self,
        conversation_id: int,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        添加对话 Embedding
        
        Args:
            conversation_id: 对话ID
            text: 对话文本
            embedding: 对话向量 (1024,)
            metadata: 元数据（可选）
        """
        current_timestamp = datetime.now().timestamp()  # Unix 时间戳（数字）
        
        payload = {
            "conversation_id": conversation_id,
            "text": text,
            "project_id": self.project_id,
            "created_at": current_timestamp,  # 存储为数字（Unix 时间戳）
            "created_at_iso": datetime.now().isoformat()  # 保留 ISO 格式用于显示
        }
        
        # 如果有 metadata，合并到 payload
        if metadata:
            payload.update(metadata)
        
        self.client.upsert(
            collection_name=f"{self.project_id}_conversations",
            points=[PointStruct(
                id=conversation_id,
                vector=embedding.tolist(),
                payload=payload
            )]
        )
    
    def conversation_exists(self, conversation_id: int) -> bool:
        """检查对话是否已存在"""
        try:
            result = self.client.retrieve(
                collection_name=f"{self.project_id}_conversations",
                ids=[conversation_id]
            )
            return len(result) > 0
        except Exception:
            return False
    
    def update_conversation_indexed(self, conversation_id: int, indexed: bool = True) -> None:
        """
        更新conversation的indexed状态
        
        Args:
            conversation_id: 对话ID
            indexed: 是否已索引
        """
        try:
            self.client.set_payload(
                collection_name=f"{self.project_id}_conversations",
                payload={"indexed": 1 if indexed else 0},
                points=[conversation_id]
            )
        except Exception as e:
            print(f"  ⚠️  更新 Qdrant conversation indexed 状态失败: {e}")
    
    def search_conversations(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_turn: Optional[int] = None,
        max_turn: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        搜索相似对话（支持时间维度和 Turn 范围过滤）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            where: 元数据过滤条件（如 {'session_id': 'xxx'}）
            start_time: 开始时间（ISO 格式）
            end_time: 结束时间（ISO 格式）
            min_turn: 最小 Turn（硬过滤）
            max_turn: 最大 Turn（硬过滤）
        """
        from qdrant_client.models import Range
        
        # 构建 Qdrant 过滤器
        conditions = []
        
        if where:
            for key, value in where.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        # 添加时间范围过滤（硬过滤）
        if start_time:
            try:
                start_timestamp = datetime.fromisoformat(start_time).timestamp()
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=Range(gte=start_timestamp)
                    )
                )
            except Exception as e:
                print(f"  ⚠️  start_time 格式错误: {e}")
        
        if end_time:
            try:
                end_timestamp = datetime.fromisoformat(end_time).timestamp()
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=Range(lte=end_timestamp)
                    )
                )
            except Exception as e:
                print(f"  ⚠️  end_time 格式错误: {e}")
        
        # 添加 Turn 范围过滤（硬过滤）
        if min_turn is not None or max_turn is not None:
            turn_range = {}
            if min_turn is not None:
                turn_range['gte'] = min_turn
            if max_turn is not None:
                turn_range['lte'] = max_turn
            
            conditions.append(
                FieldCondition(
                    key="turn",
                    range=Range(**turn_range)
                )
            )
        
        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=f"{self.project_id}_conversations",
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        ).points
        
        return {
            'ids': [hit.id for hit in results],
            'scores': [hit.score for hit in results],
            'metadatas': [hit.payload for hit in results],
            'texts': [hit.payload.get('text', '') for hit in results]
        }
    
    # ===== Community 操作 =====
    
    def add_communities(
        self,
        community_ids: List[int],
        titles: List[str],
        summaries: List[str],
        embeddings: np.ndarray
    ) -> None:
        """
        批量添加 Communities
        
        Args:
            community_ids: 社区ID列表
            titles: 标题列表
            summaries: 摘要列表
            embeddings: Embedding 矩阵 (N, 1024)
        """
        points = []
        for i, (cid, title, summary) in enumerate(zip(community_ids, titles, summaries)):
            points.append(PointStruct(
                id=cid,
                vector=embeddings[i].tolist(),
                payload={
                    "community_id": cid,
                    "title": title,
                    "summary": summary,
                    "project_id": self.project_id
                }
            ))
        
        self.client.upsert(
            collection_name=f"{self.project_id}_communities",
            points=points
        )
    
    def add_community(
        self,
        community_id: int,
        title: str,
        summary: str,
        embedding: np.ndarray
    ) -> None:
        """
        添加单个 Community
        
        Args:
            community_id: 社区ID
            title: 标题
            summary: 摘要
            embedding: Embedding 向量 (1024,)
        """
        self.client.upsert(
            collection_name=f"{self.project_id}_communities",
            points=[PointStruct(
                id=community_id,
                vector=embedding.tolist(),
                payload={
                    "community_id": community_id,
                    "title": title,
                    "summary": summary,
                    "project_id": self.project_id
                }
            )]
        )
    
    def search_communities(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """搜索相似社区"""
        # 构建 Qdrant 过滤器
        query_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=f"{self.project_id}_communities",
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        ).points
        
        return {
            'ids': [hit.id for hit in results],
            'scores': [hit.score for hit in results],  # Qdrant 返回的是相似度分数（0-1，越大越相似）
            'metadatas': [hit.payload for hit in results],
            'summaries': [hit.payload.get('summary', '') for hit in results]
        }
    
    # ===== 混合检索 =====
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        top_k_facts: int = 10,
        top_k_conversations: int = 3,
        top_k_communities: int = 2,
        conversation_filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        三级混合检索
        
        Args:
            query_embedding: 查询向量
            top_k_facts: Fact 返回数量
            top_k_conversations: Conversation 返回数量
            top_k_communities: Community 返回数量
            conversation_filter: 对话过滤条件
        
        Returns:
            {
                'facts': {...},
                'conversations': {...},
                'communities': {...}
            }
        """
        return {
            'facts': self.search_facts(query_embedding, top_k_facts, conversation_filter),
            'conversations': self.search_conversations(query_embedding, top_k_conversations),
            'communities': self.search_communities(query_embedding, top_k_communities)
        }
    
    # ===== 统计信息 =====
    
    def get_statistics(self) -> Dict[str, int]:
        """获取存储统计信息"""
        facts_info = self.client.get_collection(f"{self.project_id}_facts")
        conversations_info = self.client.get_collection(f"{self.project_id}_conversations")
        communities_info = self.client.get_collection(f"{self.project_id}_communities")
        
        return {
            'facts_count': facts_info.points_count,
            'conversations_count': conversations_info.points_count,
            'communities_count': communities_info.points_count
        }
    
    # ===== 数据管理 =====
    
    def reset(self) -> None:
        """重置所有数据（谨慎使用）"""
        # 删除所有 collections
        for suffix in ['facts', 'conversations', 'communities']:
            collection_name = f"{self.project_id}_{suffix}"
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass  # 忽略不存在的集合
        
        # 重新创建
        self._ensure_collections_exist()
    
    # ===== Hybrid Topic 相关方法 =====
    
    def _ensure_collection(self, collection_name: str, vector_size: int):
        """确保指定 Collection 存在"""
        existing = {col.name for col in self.client.get_collections().collections}
        
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
    
    async def search_topics(
        self,
        query_vector: np.ndarray,
        project_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        在 topics 集合中搜索
        
        Returns:
            [
                {
                    "topic_id": int,
                    "score": float,  # 余弦相似度
                    "top_entities": [str],
                    "fact_count": int
                }
            ]
        """
        collection_name = f"topics_{project_id}"
        
        try:
            # 确保集合存在
            self._ensure_collection(collection_name, len(query_vector))

            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=top_k
            ).points
            
            return [{
                "topic_id": hit.id,
                "score": hit.score,  # Qdrant 的 score 已经是余弦相似度
                "top_entities": hit.payload.get("top_entities", []),
                "fact_count": hit.payload.get("fact_count", 0)
            } for hit in results]
        except Exception as e:
            print(f"  ⚠️ 搜索 Topics 失败: {e}")
            return []
    
    async def get_topic_by_id(
        self,
        topic_id: int,
        project_id: str
    ) -> Dict[str, Any]:
        """获取 Topic 详细信息"""
        collection_name = f"topics_{project_id}"
        
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[topic_id],
                with_vectors=True  # 🔥 关键修复：必须显式请求向量
            )
            
            if result:
                point = result[0]
                return {
                    "topic_id": point.id,
                    "centroid": np.array(point.vector) if point.vector else None,
                    "fact_count": point.payload.get("fact_count", 0),
                    "top_entities": point.payload.get("top_entities", []),
                    "entity_counter": point.payload.get("entity_counter", {})
                }
            else:
                raise ValueError(f"Topic {topic_id} not found")
        except Exception as e:
            raise ValueError(f"获取 Topic {topic_id} 失败: {e}")
    
    async def update_topic(
        self,
        topic_id: int,
        project_id: str,
        centroid: np.ndarray,
        fact_count: int,
        top_entities: List[str],
        entity_counter: Dict[str, int]
    ):
        """更新 Topic"""
        collection_name = f"topics_{project_id}"
        
        self.client.upsert(
            collection_name=collection_name,
            points=[PointStruct(
                id=topic_id,
                vector=centroid.tolist(),
                payload={
                    "project_id": project_id,
                    "fact_count": fact_count,
                    "top_entities": top_entities,
                    "entity_counter": entity_counter
                }
            )]
        )
    
    async def add_to_buffer(
        self,
        facts: List[Dict[str, Any]],
        project_id: str
    ):
        """添加到 Buffer 集合"""
        collection_name = f"buffer_{project_id}"
        
        if not facts:
            return
        
        # 确保集合存在
        self._ensure_collection(collection_name, len(facts[0]["vector"]))
        
        points = [PointStruct(
            id=fact["id"],
            vector=fact["vector"].tolist(),
            payload={
                "entities": fact["entities"],
                "content": fact["content"],
                "status": "BUFFER"
            }
        ) for fact in facts]
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    async def get_fact_vector_from_buffer(
        self,
        fact_id: int,
        project_id: str
    ) -> np.ndarray:
        """从 Buffer 获取 Fact 向量"""
        collection_name = f"buffer_{project_id}"
        
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[fact_id]
            )
            
            if result:
                return np.array(result[0].vector)
            else:
                raise ValueError(f"Fact {fact_id} not found in buffer")
        except Exception as e:
            raise ValueError(f"从 Buffer 获取 Fact {fact_id} 失败: {e}")
    
    async def create_topic(
        self,
        project_id: str,
        centroid: np.ndarray,
        fact_count: int,
        top_entities: List[str],
        entity_counter: Dict[str, int],
        title: str,
        summary: str
    ) -> int:
        """创建新 Topic，返回 topic_id"""
        collection_name = f"topics_{project_id}"
        
        # 确保集合存在
        self._ensure_collection(collection_name, len(centroid))
        
        # 生成新 ID（使用时间戳 + 随机数）
        import time
        import random
        topic_id = int(time.time() * 1000) % 1000000 + random.randint(0, 999)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[PointStruct(
                id=topic_id,
                vector=centroid.tolist(),
                payload={
                    "project_id": project_id,
                    "fact_count": fact_count,
                    "top_entities": top_entities,
                    "entity_counter": entity_counter,
                    "title": title,
                    "summary": summary,
                    "created_at": datetime.now().isoformat()
                }
            )]
        )
        
        return topic_id
    
    async def remove_from_buffer(
        self,
        fact_ids: List[int],
        project_id: str
    ):
        """从 Buffer 删除"""
        collection_name = f"buffer_{project_id}"
        
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=fact_ids
            )
        except Exception as e:
            print(f"  ⚠️ 从 Buffer 删除失败: {e}")


# ===== 使用示例 =====

if __name__ == "__main__":
    """
    使用示例 - 基于 Qdrant
    """
    from pathlib import Path
    import numpy as np
    
    # 初始化
    store = GauzRagVectorStore(
        persist_directory=Path("./qdrant_db"),
        project_id="test_project",
        vector_dim=1024
    )
    
    # 添加 Facts
    fact_ids = [1, 2, 3]
    contents = [
        "蓝星科技是一家新能源公司",
        "太阳帆计划由张博士领导",
        "王芳负责数据分析"
    ]
    embeddings = np.random.rand(3, 1024)  # 实际使用真实 embeddings
    conversation_ids = [1, 1, 1]
    
    store.add_facts(fact_ids, contents, embeddings, conversation_ids)
    
    # 搜索
    query_emb = np.random.rand(1024)
    results = store.search_facts(query_emb, top_k=2)
    print(f"找到 {len(results['ids'])} 条相似 facts")
    
    # 添加对话
    conv_emb = np.random.rand(1024)
    store.add_conversation(1, "完整对话文本...", conv_emb)
    
    # 混合检索
    hybrid_results = store.hybrid_search(query_emb)
    print(f"Facts: {len(hybrid_results['facts']['ids'])}")
    print(f"Conversations: {len(hybrid_results['conversations']['ids'])}")
    
    # 统计
    stats = store.get_statistics()
    print(f"统计: {stats}")
    print("\n=== Qdrant 向量数据库 ===")
    print(f"持久化目录: {store.persist_directory}")
    print(f"向量维度: {store.vector_dim}")

