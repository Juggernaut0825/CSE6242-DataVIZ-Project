"""
Buffer 整合器：定期扫描并合并滞留在 Buffer 中的相似 Facts

设计原理：
- Buffer 是 per-batch 的，不同批次的 Facts 不会自动合并
- 这个工具提供一个全局视角，定期扫描所有 Buffer Facts
- 如果发现语义相似的 Facts 达到阈值，主动晋升为 Topic

使用场景：
- 作为定时任务（Cron Job）运行
- 或者在每次 extract 后，检查 Buffer 大小，超过阈值时触发
"""

import asyncio
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np


class BufferConsolidator:
    """Buffer 整合器"""
    
    def __init__(
        self,
        vector_store,
        neo4j_store,
        hybrid_detector,
        project_id: str,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 3
    ):
        """
        初始化整合器
        
        Args:
            vector_store: Qdrant 存储
            neo4j_store: Neo4j 存储
            hybrid_detector: HybridTopicDetector 实例
            project_id: 项目 ID
            similarity_threshold: 相似度阈值
            min_cluster_size: 最小聚类大小
        """
        self.vector_store = vector_store
        self.neo4j_store = neo4j_store
        self.detector = hybrid_detector
        self.project_id = project_id
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
    
    async def consolidate(self):
        """
        执行整合
        
        流程：
        1. 从 Qdrant 获取所有 Buffer Facts（含向量）
        2. 计算相似度矩阵，构建相似图
        3. 使用连通分量算法聚类
        4. 对每个聚类，调用 HybridTopicDetector 的晋升逻辑
        """
        print(f"\n{'='*60}")
        print(f"[Buffer 整合] 开始扫描 project_id={self.project_id}")
        print(f"{'='*60}")
        
        # 1. 获取所有 Buffer Facts
        buffer_facts = await self._get_buffer_facts()
        
        if len(buffer_facts) < self.min_cluster_size:
            print(f"  ℹ️  Buffer 中只有 {len(buffer_facts)} 条 Facts，低于阈值 {self.min_cluster_size}，跳过")
            return
        
        print(f"  📦 Buffer 中共 {len(buffer_facts)} 条 Facts")
        
        # 2. 聚类
        clusters = self._cluster_facts(buffer_facts)
        
        print(f"  🔍 发现 {len(clusters)} 个潜在聚类")
        
        # 3. 晋升符合条件的聚类
        promoted_count = 0
        for i, cluster in enumerate(clusters):
            if len(cluster) >= self.min_cluster_size:
                print(f"\n  [聚类 {i+1}] 包含 {len(cluster)} 条 Facts，尝试晋升...")
                try:
                    await self._promote_cluster(cluster)
                    promoted_count += 1
                    print(f"    ✅ 晋升成功")
                except Exception as e:
                    print(f"    ❌ 晋升失败: {e}")
        
        print(f"\n{'='*60}")
        print(f"[Buffer 整合] 完成！共晋升 {promoted_count} 个 Topics")
        print(f"{'='*60}\n")
    
    async def _get_buffer_facts(self) -> List[Dict[str, Any]]:
        """从 Qdrant 获取所有 Buffer Facts（含向量）"""
        collection_name = f"buffer_{self.project_id}"
        
        try:
            # Qdrant 的 scroll 方法可以批量获取所有点
            results = self.vector_store.client.scroll(
                collection_name=collection_name,
                limit=1000,  # 每批最多 1000
                with_vectors=True,
                with_payload=True
            )
            
            facts = []
            for point in results[0]:
                facts.append({
                    'fact_id': point.payload['fact_id'],
                    'content': point.payload['content'],
                    'entities': point.payload.get('entities', []),
                    'vector': np.array(point.vector)
                })
            
            return facts
        except Exception as e:
            print(f"  ⚠️ 获取 Buffer Facts 失败: {e}")
            return []
    
    def _cluster_facts(self, facts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """使用相似度聚类"""
        if not facts:
            return []
        
        # 构建相似度矩阵
        n = len(facts)
        adjacency = [[False] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # 计算余弦相似度
                v1 = facts[i]['vector']
                v2 = facts[j]['vector']
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                if similarity >= self.similarity_threshold:
                    adjacency[i][j] = True
                    adjacency[j][i] = True
        
        # Union-Find 聚类
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i][j]:
                    union(i, j)
        
        # 构建聚类
        clusters_dict = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters_dict[root].append(facts[i])
        
        return list(clusters_dict.values())
    
    async def _promote_cluster(self, cluster: List[Dict[str, Any]]):
        """将一个聚类晋升为 Topic"""
        # 转换为 HybridTopicDetector 的格式
        batch_facts = []
        for fact in cluster:
            batch_facts.append({
                "fact_id": fact['fact_id'],
                "content": fact['content'],
                "entities": fact['entities'],
                "vector": fact['vector'].tolist()
            })
        
        # 调用 Detector 的 process_batch（应该会触发晋升）
        result = await self.detector.process_batch(batch_facts)
        
        if result['action'] != 'promote':
            raise Exception(f"预期晋升，但实际动作是: {result['action']}")


# ========== 使用示例 ==========

async def run_consolidation_for_project(project_id: str):
    """为指定项目运行 Buffer 整合"""
    from .config import RAGConfig
    from .vector_store import GauzRagVectorStore
    from .neo4j_storage import Neo4jStorage
    from .hybrid_topic_detector import HybridTopicDetector
    from .embedder import DashScopeEmbedder
    from openai import AsyncOpenAI
    import os
    
    config = RAGConfig()
    
    # 初始化依赖
    vector_store = GauzRagVectorStore(
        persist_directory=config.output_dir / "qdrant_db",
        project_id=project_id,
        use_server=True,
        server_url=config.qdrant_url
    )
    
    neo4j_store = Neo4jStorage(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password
    )
    
    embedder = DashScopeEmbedder()
    
    llm_client = AsyncOpenAI(
        api_key=os.getenv("GAUZ_LLM_API_KEY"),
        base_url=os.getenv("GAUZ_LLM_API_BASE")
    )
    
    detector = HybridTopicDetector(
        vector_store=vector_store,
        neo4j_store=neo4j_store,
        embedder=embedder,
        llm_client=llm_client,
        llm_model=os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini"),
        project_id=project_id
    )
    
    # 运行整合
    consolidator = BufferConsolidator(
        vector_store=vector_store,
        neo4j_store=neo4j_store,
        hybrid_detector=detector,
        project_id=project_id
    )
    
    await consolidator.consolidate()
    
    # 关闭连接
    vector_store.close()
    await neo4j_store.close()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(run_consolidation_for_project("hybrid_topic_test"))
