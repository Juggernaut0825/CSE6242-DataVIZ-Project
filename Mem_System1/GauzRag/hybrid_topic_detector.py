"""
Hybrid Topic Detector - 混合式 Topic 检测器
结合向量相似度和实体共现的 Topic 聚类方案

核心流程：
1. 输入标准化：接收 Batch_Facts（一轮对话的多个 Facts）
2. 聚合计算：计算批次质心（L2归一化）+ 聚合实体集合
3. 混合匹配：向量粗筛 + 实体加权（Entity Boost）
4. 决策与归档：命中归入/未命中进Buffer + 连通性检查
"""
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from collections import Counter
from dataclasses import dataclass
import asyncio


@dataclass
class BatchFact:
    """单个 Fact 的数据结构"""
    fact_id: int
    vector: np.ndarray
    entities: List[str]
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TopicCandidate:
    """Topic 候选项"""
    topic_id: int
    vector_score: float  # 向量相似度分数
    entity_boost: float  # 实体加权分数
    final_score: float   # 最终分数
    top_entities: List[str]
    fact_count: int


class HybridTopicDetector:
    """
    混合式 Topic 检测器
    
    优势：
    1. 向量 + 实体混合匹配（精度高）
    2. Buffer 机制延迟决策（避免过早分裂）
    3. 图连通性检查（利用 Neo4j 实体边）
    4. 数学严谨（L2归一化 + 加权移动平均）
    """
    
    def __init__(
        self,
        vector_store,  # GauzRagVectorStore 实例
        neo4j_store,   # Neo4jEntityStore 实例
        embedder,      # DashScopeEmbedder 实例
        llm_client,    # OpenAI 客户端
        llm_model: str,
        project_id: str,
        # 超参数
        vector_top_k: int = 5,
        entity_boost_weight: float = 0.05,
        max_entity_boost: float = 0.20,
        match_threshold: float = 0.75,
        buffer_promote_threshold: int = 3,
        max_buffer_size: int = 100
    ):
        """
        初始化混合检测器
        
        Args:
            vector_store: Qdrant 向量存储
            neo4j_store: Neo4j 图存储
            embedder: Embedding 模型
            llm_client: LLM 客户端（用于生成 Topic Summary）
            llm_model: LLM 模型名称
            project_id: 项目 ID
            vector_top_k: 向量搜索返回的候选数量（默认5）
            entity_boost_weight: 每个共享实体的奖励分（默认0.05）
            max_entity_boost: 实体奖励分上限（默认0.20）
            match_threshold: 匹配阈值（默认0.75）
            buffer_promote_threshold: Buffer晋升阈值（默认3个Facts）
            max_buffer_size: Buffer最大容量（默认100）
        """
        self.vector_store = vector_store
        self.neo4j_store = neo4j_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.project_id = project_id
        
        # 超参数
        self.vector_top_k = vector_top_k
        self.entity_boost_weight = entity_boost_weight
        self.max_entity_boost = max_entity_boost
        self.match_threshold = match_threshold
        self.buffer_promote_threshold = buffer_promote_threshold
        self.max_buffer_size = max_buffer_size
    
    # ========== 🟢 第一步：输入标准化 ==========
    
    def standardize_input(
        self,
        facts: List[Dict[str, Any]]
    ) -> List[BatchFact]:
        """
        标准化输入数据
        
        Args:
            facts: [
                {
                    "fact_id": int,
                    "content": str,
                    "entities": [str],
                    "vector": np.ndarray (optional)
                },
                ...
            ]
        
        Returns:
            List[BatchFact]
        """
        batch_facts = []
        
        for fact in facts:
            # 如果没有向量，现场生成
            if "vector" not in fact or fact["vector"] is None:
                vector = self.embedder.encode([fact["content"]])[0]
            else:
                vector = np.array(fact["vector"])
            
            batch_facts.append(BatchFact(
                fact_id=fact["fact_id"],
                vector=vector,
                entities=fact.get("entities", []),
                content=fact["content"],
                metadata=fact.get("metadata")
            ))
        
        return batch_facts
    
    # ========== 🔵 第二步：聚合计算 ==========
    
    def aggregate_batch(
        self,
        batch_facts: List[BatchFact]
    ) -> Tuple[np.ndarray, Set[str]]:
        """
        聚合计算批次特征
        
        1. 计算批次质心（L2归一化）
        2. 聚合实体集合
        
        Args:
            batch_facts: 标准化后的 Facts
        
        Returns:
            (V_final, E_batch)
            - V_final: 归一化后的批次质心向量
            - E_batch: 实体集合
        """
        # 1. 计算批次质心
        vectors = np.array([f.vector for f in batch_facts])
        V_batch = np.mean(vectors, axis=0)
        
        # L2 归一化（保证在单位球面上）
        norm = np.linalg.norm(V_batch)
        if norm > 0:
            V_final = V_batch / norm
        else:
            V_final = V_batch
        
        # 2. 聚合实体集合（去重）
        E_batch = set()
        for fact in batch_facts:
            E_batch.update(fact.entities)
        
        return V_final, E_batch
    
    # ========== 🟡 第三步：混合匹配 ==========
    
    async def hybrid_match(
        self,
        V_final: np.ndarray,
        E_batch: Set[str]
    ) -> List[TopicCandidate]:
        """
        混合匹配：向量粗筛 + 实体加权
        
        Args:
            V_final: 归一化后的批次质心
            E_batch: 实体集合
        
        Returns:
            按 final_score 降序排列的候选 Topics
        """
        # 1. 向量粗筛（Qdrant）
        vector_results = await self._vector_search_topics(V_final)
        
        if not vector_results:
            return []
        
        # 2. 实体加权
        candidates = []
        for result in vector_results:
            topic_id = result['topic_id']
            S_vec = result['score']
            top_entities = set(result.get('top_entities', []))
            
            # 计算 Entity Boost
            N_shared = len(E_batch & top_entities)
            Boost = min(N_shared * self.entity_boost_weight, self.max_entity_boost)
            
            S_final = S_vec + Boost
            
            candidates.append(TopicCandidate(
                topic_id=topic_id,
                vector_score=S_vec,
                entity_boost=Boost,
                final_score=S_final,
                top_entities=list(top_entities),
                fact_count=result.get('fact_count', 0)
            ))
        
        # 3. 按 final_score 降序排序
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        
        return candidates
    
    async def _vector_search_topics(
        self,
        query_vector: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        在 Qdrant 的 topics 集合中搜索
        
        Returns:
            [
                {
                    "topic_id": int,
                    "score": float,
                    "top_entities": [str],
                    "fact_count": int
                }
            ]
        """
        try:
            # 假设 vector_store 有 search_topics 方法
            results = await self.vector_store.search_topics(
                query_vector=query_vector,
                project_id=self.project_id,
                top_k=self.vector_top_k
            )
            return results
        except Exception as e:
            print(f"  ⚠️ 向量搜索失败: {e}")
            return []
    
    # ========== 🔴 第四步：决策与归档 ==========
    
    async def decide_and_archive(
        self,
        batch_facts: List[BatchFact],
        candidates: List[TopicCandidate],
        V_final: np.ndarray,
        E_batch: Set[str]
    ) -> Dict[str, Any]:
        """
        决策与归档
        
        Args:
            batch_facts: 当前 Batch 的 Facts
            candidates: 匹配的候选 Topics
            V_final: 批次质心
            E_batch: 实体集合
        
        Returns:
            {
                "action": "assign" | "buffer" | "promote",
                "topic_id": int (if assign or promote),
                "details": {...}
            }
        """
        if not candidates:
            # 没有候选，直接进 Buffer
            return await self._handle_buffer(batch_facts, V_final, E_batch)
        
        best_candidate = candidates[0]
        
        if best_candidate.final_score >= self.match_threshold:
            # 🚩 分支 A：命中
            return await self._handle_assign(
                batch_facts, best_candidate, V_final, E_batch
            )
        else:
            # 🚩 分支 B：未命中
            return await self._handle_buffer(batch_facts, V_final, E_batch)
    
    async def _handle_assign(
        self,
        batch_facts: List[BatchFact],
        topic: TopicCandidate,
        V_final: np.ndarray,
        E_batch: Set[str]
    ) -> Dict[str, Any]:
        """
        处理命中：归入现有 Topic
        
        步骤：
        1. 更新 Topic 质心（加权移动平均）
        2. 更新 Topic 实体计数器
        3. 更新 Qdrant 中的 Topic
        4. 建立 Neo4j 连接
        """
        print(f"  ✅ 命中 Topic {topic.topic_id} (分数: {topic.final_score:.3f})")
        
        # 1. 获取旧的 Topic 信息
        old_topic_info = await self._get_topic_info(topic.topic_id)
        
        # 🔥 防御性检查：centroid 可能为 None
        if old_topic_info.get('centroid') is None:
            print(f"  ⚠️ Topic {topic.topic_id} 缺少 centroid，无法更新！回退到 Buffer")
            return await self.buffer_action(batch_facts, V_final, E_batch)
        
        V_old = np.array(old_topic_info['centroid'])
        N_old = old_topic_info['fact_count']
        entity_counter = Counter(old_topic_info.get('entity_counter', {}))
        
        # 2. 更新质心（加权移动平均）
        N_batch = len(batch_facts)
        V_new = (V_old * N_old + V_final * N_batch) / (N_old + N_batch)
        
        # L2 归一化
        norm = np.linalg.norm(V_new)
        if norm > 0:
            V_new = V_new / norm
        
        # 3. 更新实体计数器
        for entity in E_batch:
            entity_counter[entity] += 1
        
        # 4. 获取 top_entities（频率最高的前10个）
        top_entities = [e for e, _ in entity_counter.most_common(10)]
        
        # 5. 更新 Qdrant
        await self._update_topic_in_qdrant(
            topic_id=topic.topic_id,
            centroid=V_new,
            fact_count=N_old + N_batch,
            top_entities=top_entities,
            entity_counter=dict(entity_counter)
        )
        
        # 6. 建立 Neo4j 连接
        await self._link_facts_to_topic(
            fact_ids=[f.fact_id for f in batch_facts],
            topic_id=topic.topic_id
        )
        
        return {
            "action": "assign",
            "topic_id": topic.topic_id,
            "details": {
                "final_score": topic.final_score,
                "vector_score": topic.vector_score,
                "entity_boost": topic.entity_boost,
                "fact_count": N_old + N_batch
            }
        }
    
    async def _handle_buffer(
        self,
        batch_facts: List[BatchFact],
        V_final: np.ndarray,
        E_batch: Set[str]
    ) -> Dict[str, Any]:
        """
        处理未命中：进入 Buffer + 连通性检查
        
        步骤：
        1. 存入 Qdrant Buffer 集合
        2. 标记 Neo4j 状态为 'BUFFER'
        3. 连通性检查（查询共享实体的 Buffer Facts）
        4. 二次决策（晋升 or 等待）
        """
        print(f"  ⚠️ 未命中任何 Topic，进入 Buffer")
        
        # 1. 存入 Buffer（Qdrant）
        await self._add_to_buffer_qdrant(batch_facts)
        
        # 2. 标记状态（Neo4j）
        await self._mark_buffer_status(
            fact_ids=[f.fact_id for f in batch_facts]
        )
        
        # 3. 连通性检查
        found_neighbors = await self._check_buffer_connectivity(
            batch_fact_ids=[f.fact_id for f in batch_facts]
        )
        
        # 4. 二次决策
        N_total = len(batch_facts) + len(found_neighbors)
        
        if N_total >= self.buffer_promote_threshold:
            # 🔥 立即晋升
            print(f"  🔥 Buffer 连通性达标 (N={N_total})，立即晋升！")
            
            return await self._promote_from_buffer(
                batch_facts, found_neighbors, V_final, E_batch
            )
        else:
            # 🧊 保持等待
            print(f"  🧊 Buffer 等待中 (N={N_total} < {self.buffer_promote_threshold})")
            
            return {
                "action": "buffer",
                "topic_id": None,
                "details": {
                    "buffer_size": N_total,
                    "neighbors_found": len(found_neighbors)
                }
            }
    
    async def _promote_from_buffer(
        self,
        batch_facts: List[BatchFact],
        neighbor_ids: List[int],
        V_final: np.ndarray,
        E_batch: Set[str]
    ) -> Dict[str, Any]:
        """
        从 Buffer 晋升：创建新 Topic
        
        步骤：
        1. 获取邻居 Facts 的详细信息
        2. 计算新 Topic 的质心
        3. 聚合实体集合
        4. 生成 Topic Summary（LLM）
        5. 创建新 Topic（Qdrant + Neo4j）
        6. 从 Buffer 删除
        """
        # 1. 获取邻居 Facts
        neighbor_facts = await self._get_facts_by_ids(neighbor_ids)
        
        all_facts = batch_facts + neighbor_facts
        
        # 2. 计算新 Topic 质心
        all_vectors = np.array([f.vector for f in all_facts])
        new_centroid = np.mean(all_vectors, axis=0)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        
        # 3. 聚合实体集合
        entity_counter = Counter()
        for fact in all_facts:
            entity_counter.update(fact.entities)
        
        top_entities = [e for e, _ in entity_counter.most_common(10)]
        
        # 4. 生成 Topic Summary（LLM）
        summary = await self._generate_topic_summary(
            facts=[f.content for f in all_facts],
            top_entities=top_entities
        )
        
        # 5. 创建新 Topic
        new_topic_id = await self._create_new_topic(
            centroid=new_centroid,
            fact_count=len(all_facts),
            top_entities=top_entities,
            entity_counter=dict(entity_counter),
            title=summary['title'],
            summary_text=summary['summary']
        )
        
        # 6. 建立连接
        await self._link_facts_to_topic(
            fact_ids=[f.fact_id for f in all_facts],
            topic_id=new_topic_id
        )
        
        # 7. 从 Buffer 删除
        await self._remove_from_buffer(
            fact_ids=[f.fact_id for f in all_facts]
        )
        
        print(f"  ✨ 创建新 Topic {new_topic_id} (包含 {len(all_facts)} Facts)")
        
        return {
            "action": "promote",
            "topic_id": new_topic_id,
            "details": {
                "fact_count": len(all_facts),
                "from_batch": len(batch_facts),
                "from_buffer": len(neighbor_facts),
                "title": summary['title']
            }
        }
    
    # ========== 辅助方法 ==========
    
    async def _get_topic_info(self, topic_id: int) -> Dict[str, Any]:
        """从 Qdrant 获取 Topic 信息"""
        # 实现：查询 Qdrant topics 集合
        # 这里需要 vector_store 提供 get_topic_by_id 方法
        return await self.vector_store.get_topic_by_id(topic_id, self.project_id)
    
    async def _update_topic_in_qdrant(
        self,
        topic_id: int,
        centroid: np.ndarray,
        fact_count: int,
        top_entities: List[str],
        entity_counter: Dict[str, int]
    ):
        """更新 Qdrant 中的 Topic"""
        await self.vector_store.update_topic(
            topic_id=topic_id,
            project_id=self.project_id,
            centroid=centroid,
            fact_count=fact_count,
            top_entities=top_entities,
            entity_counter=entity_counter
        )
    
    async def _link_facts_to_topic(
        self,
        fact_ids: List[int],
        topic_id: int
    ):
        """在 Neo4j 中建立 Fact → Topic 连接"""
        await self.neo4j_store.link_facts_to_topic(
            fact_ids=fact_ids,
            topic_id=topic_id,
            project_id=self.project_id
        )
    
    async def _add_to_buffer_qdrant(self, batch_facts: List[BatchFact]):
        """将 Facts 添加到 Qdrant Buffer 集合"""
        await self.vector_store.add_to_buffer(
            facts=[{
                "id": f.fact_id,
                "vector": f.vector,
                "entities": f.entities,
                "content": f.content
            } for f in batch_facts],
            project_id=self.project_id
        )
    
    async def _mark_buffer_status(self, fact_ids: List[int]):
        """在 Neo4j 中标记 Facts 状态为 BUFFER"""
        await self.neo4j_store.mark_facts_as_buffer(
            fact_ids=fact_ids,
            project_id=self.project_id
        )
    
    async def _check_buffer_connectivity(
        self,
        batch_fact_ids: List[int]
    ) -> List[int]:
        """
        检查 Buffer 连通性（Neo4j Cypher）
        
        查询：与当前 Batch 共享实体的 Buffer Facts
        """
        return await self.neo4j_store.find_connected_buffer_facts(
            fact_ids=batch_fact_ids,
            project_id=self.project_id
        )
    
    async def _get_facts_by_ids(self, fact_ids: List[int]) -> List[BatchFact]:
        """根据 ID 获取 Facts 详细信息"""
        facts_data = await self.neo4j_store.get_facts_by_ids(
            fact_ids=fact_ids,
            project_id=self.project_id
        )
        
        batch_facts = []
        for data in facts_data:
            # 从 Qdrant Buffer 获取向量
            vector = await self.vector_store.get_fact_vector_from_buffer(
                fact_id=data['fact_id'],
                project_id=self.project_id
            )
            
            batch_facts.append(BatchFact(
                fact_id=data['fact_id'],
                vector=vector,
                entities=data.get('entities', []),
                content=data['content']
            ))
        
        return batch_facts
    
    async def _generate_topic_summary(
        self,
        facts: List[str],
        top_entities: List[str]
    ) -> Dict[str, str]:
        """使用 LLM 生成 Topic Summary"""
        facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
        entities_text = ", ".join(top_entities[:5])
        
        prompt = f"""你是一个知识图谱专家。以下是一组语义相关的事实（Facts），请生成一个简洁的 Topic。

【主要实体】
{entities_text}

【事实列表】
{facts_text}

请生成：
1. title: Topic 标题（5-10字，概括核心主题）
2. summary: Topic 摘要（100-150字，整合所有事实的共同主题）

注意：这些事实通过向量相似度和实体共现被聚在一起，它们一定有共同的主题。

返回 JSON 格式。
"""
        
        import json
        import re
        
        # Retry 机制（最多 3 次）
        for attempt in range(3):
            try:
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "你是一个知识总结专家。必须返回有效的 JSON 格式。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    timeout=10.0  # 10秒超时
                )
                
                content = response.choices[0].message.content.strip()
                
                # 清洗 Markdown 代码块（```json ... ```）
                if content.startswith('```'):
                    # 提取 ```json 和 ``` 之间的内容
                    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                
                # 尝试解析 JSON
                result = json.loads(content)
                
                # 验证必需字段
                if 'title' in result and result['title']:
                    return {
                        'title': result.get('title', 'Untitled Topic'),
                        'summary': result.get('summary', '')
                    }
                else:
                    raise ValueError("JSON 缺少 title 字段")
                    
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Attempt {attempt + 1}/3: JSON 解析失败 - {e}")
                if attempt == 2:  # 最后一次尝试
                    print(f"     原始响应: {content[:200]}...")
            except Exception as e:
                print(f"  ⚠️ Attempt {attempt + 1}/3: 生成 Summary 失败 - {e}")
        
        # 所有尝试失败，返回 Fallback
        print(f"  💀 所有尝试失败，使用默认标题")
        return {
            'title': f'Topic ({len(facts)} facts)',
            'summary': f'包含 {len(facts)} 个相关事实'
        }
    
    async def _create_new_topic(
        self,
        centroid: np.ndarray,
        fact_count: int,
        top_entities: List[str],
        entity_counter: Dict[str, int],
        title: str,
        summary_text: str
    ) -> int:
        """创建新 Topic（Qdrant + Neo4j）"""
        # 1. 在 Qdrant 创建
        topic_id = await self.vector_store.create_topic(
            project_id=self.project_id,
            centroid=centroid,
            fact_count=fact_count,
            top_entities=top_entities,
            entity_counter=entity_counter,
            title=title,
            summary=summary_text
        )
        
        # 2. 在 Neo4j 创建
        await self.neo4j_store.create_topic_node(
            topic_id=topic_id,
            project_id=self.project_id,
            title=title,
            summary=summary_text
        )
        
        return topic_id
    
    async def _remove_from_buffer(self, fact_ids: List[int]):
        """从 Buffer 删除（Qdrant + Neo4j）"""
        # 1. 从 Qdrant Buffer 删除
        await self.vector_store.remove_from_buffer(
            fact_ids=fact_ids,
            project_id=self.project_id
        )
        
        # 2. 清除 Neo4j 状态标记
        await self.neo4j_store.clear_buffer_status(
            fact_ids=fact_ids,
            project_id=self.project_id
        )
    
    # ========== 主入口 ==========
    
    async def process_batch(
        self,
        facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        处理一个 Batch 的 Facts
        
        Args:
            facts: [
                {
                    "fact_id": int,
                    "content": str,
                    "entities": [str],
                    "vector": np.ndarray (optional)
                }
            ]
        
        Returns:
            {
                "action": "assign" | "buffer" | "promote",
                "topic_id": int | None,
                "details": {...}
            }
        """
        print(f"\n{'='*60}")
        print(f"处理 Batch: {len(facts)} Facts")
        print(f"{'='*60}")
        
        # 1. 输入标准化
        batch_facts = self.standardize_input(facts)
        print(f"  [1/4] 输入标准化完成")
        
        # 2. 聚合计算
        V_final, E_batch = self.aggregate_batch(batch_facts)
        print(f"  [2/4] 聚合计算完成")
        print(f"    - 质心向量: {V_final.shape}")
        print(f"    - 实体集合: {len(E_batch)} 个实体")
        
        # 3. 混合匹配
        candidates = await self.hybrid_match(V_final, E_batch)
        print(f"  [3/4] 混合匹配完成")
        if candidates:
            print(f"    - 最佳候选: Topic {candidates[0].topic_id} (分数: {candidates[0].final_score:.3f})")
            print(f"      · 向量分: {candidates[0].vector_score:.3f}")
            print(f"      · 实体加权: +{candidates[0].entity_boost:.3f}")
        else:
            print(f"    - 无候选 Topics")
        
        # 4. 决策与归档
        result = await self.decide_and_archive(
            batch_facts, candidates, V_final, E_batch
        )
        print(f"  [4/4] 决策完成: {result['action']}")
        
        return result
