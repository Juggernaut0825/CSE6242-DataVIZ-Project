"""
Semantic Topic Detector - 基于向量聚类的增量式 Topic 检测
核心思想：物以类聚 - 语义相似的 Facts 自动聚合成 Topics
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from openai import OpenAI
import json
from datetime import datetime


class SemanticTopicDetector:
    """
    基于向量的流式 Topic 聚类器
    
    优势：
    1. 增量友好：新 Fact 只需一次向量搜索即可归类
    2. 语义精准：基于内容相似度，避免连通分量的"语义漂移"问题
    3. 抗噪强：不依赖 Fact-Fact 边，容错率高
    4. LLM 友好：同 Topic 内的 Facts 语义高度一致，易于生成精准 Summary
    """
    
    def __init__(
        self,
        db_manager,
        vector_store,
        embedder,
        llm_client: OpenAI,
        llm_model: str,
        similarity_threshold: float = 0.75,  # 余弦相似度阈值
        min_topic_size: int = 3  # 最小 Topic 大小（用于过滤小簇）
    ):
        """
        初始化检测器
        
        Args:
            db_manager: 数据库管理器
            vector_store: Qdrant 向量存储
            embedder: Embedding 模型
            llm_client: OpenAI 客户端（用于生成 Topic Summary）
            llm_model: LLM 模型名称
            similarity_threshold: 相似度阈值（0.75 = 距离 0.25）
            min_topic_size: 最小 Topic 大小
        """
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold
        self.min_topic_size = min_topic_size
        
        # 距离阈值 = 1 - 余弦相似度
        # 余弦相似度 0.75 → 距离 0.25
        self.distance_threshold = 1 - similarity_threshold
    
    def assign_fact_to_topic(
        self,
        fact_id: int,
        fact_content: str,
        fact_embedding: np.ndarray,
        project_id: str,
        force_create: bool = False
    ) -> Tuple[int, bool]:
        """
        为单个 Fact 分配 Topic（增量模式）
        
        Args:
            fact_id: Fact ID
            fact_content: Fact 内容
            fact_embedding: Fact 的向量
            project_id: 项目 ID
            force_create: 是否强制创建新 Topic（忽略已有质心）
        
        Returns:
            (topic_id, is_new_topic)
        """
        if not force_create:
            # Step 1: 在 Qdrant 中搜索最近的 Topic 质心
            existing_topics = self._search_nearest_topics(
                fact_embedding,
                project_id,
                top_k=1
            )
            
            if existing_topics and existing_topics[0]['distance'] < self.distance_threshold:
                # Case A: 归并到现有 Topic
                topic_id = existing_topics[0]['topic_id']
                print(f"  Fact {fact_id} 归入 Topic {topic_id}（距离: {existing_topics[0]['distance']:.3f}）")
                
                # 更新 Topic（重新计算质心 + 触发 LLM 增量更新）
                self._add_fact_to_topic(
                    topic_id=topic_id,
                    fact_id=fact_id,
                    fact_content=fact_content,
                    fact_embedding=fact_embedding,
                    project_id=project_id
                )
                
                return topic_id, False
        
        # Case B: 创建新 Topic
        topic_id = self._create_new_topic(
            initial_fact_id=fact_id,
            initial_fact_content=fact_content,
            initial_embedding=fact_embedding,
            project_id=project_id
        )
        
        print(f"  Fact {fact_id} 创建新 Topic {topic_id}")
        return topic_id, True
    
    def _search_nearest_topics(
        self,
        query_embedding: np.ndarray,
        project_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        在 Qdrant 中搜索最近的 Topic 质心
        
        Returns:
            [{'topic_id': int, 'distance': float, 'title': str, 'summary': str}]
        """
        try:
            # 直接使用 vector_store 的 search_topics 方法
            # 注意：需要在 Qdrant 中维护一个 "topic_centroids" collection
            results = self.vector_store.search_topic_centroids(
                query_embedding,
                project_id=project_id,
                top_k=top_k
            )
            
            return results
        except Exception as e:
            print(f"  ⚠️ 搜索 Topic 质心失败: {e}")
            return []
    
    def _create_new_topic(
        self,
        initial_fact_id: int,
        initial_fact_content: str,
        initial_embedding: np.ndarray,
        project_id: str
    ) -> int:
        """
        创建新 Topic
        
        Returns:
            topic_id
        """
        # 1. 生成初始 Title 和 Summary（基于单个 Fact）
        topic_info = self._generate_topic_summary(
            fact_contents=[initial_fact_content],
            is_initial=True
        )
        
        # 2. 在 Neo4j 中创建 Topic 节点
        topic_id = self.db_manager.create_topic(
            project_id=project_id,
            title=topic_info['title'],
            summary=topic_info['summary'],
            centroid_vector=initial_embedding.tolist(),
            fact_count=1
        )
        
        # 3. 建立 Fact → Topic 关系
        self.db_manager.link_fact_to_topic(
            fact_id=initial_fact_id,
            topic_id=topic_id
        )
        
        # 4. 在 Qdrant 中添加 Topic 质心
        self.vector_store.add_topic_centroid(
            topic_id=topic_id,
            project_id=project_id,
            centroid_vector=initial_embedding,
            title=topic_info['title'],
            summary=topic_info['summary']
        )
        
        return topic_id
    
    def _add_fact_to_topic(
        self,
        topic_id: int,
        fact_id: int,
        fact_content: str,
        fact_embedding: np.ndarray,
        project_id: str
    ):
        """
        将 Fact 添加到现有 Topic，并更新质心和 Summary
        """
        # 1. 建立 Fact → Topic 关系
        self.db_manager.link_fact_to_topic(
            fact_id=fact_id,
            topic_id=topic_id
        )
        
        # 2. 获取当前 Topic 的所有 Facts
        topic_facts = self.db_manager.get_facts_in_topic(topic_id)
        
        # 3. 重新计算质心（增量式平均）
        # 方法：old_centroid * (n-1)/n + new_vector * 1/n
        old_centroid = np.array(topic_facts['centroid_vector'])
        n = len(topic_facts['fact_ids']) + 1  # 包含新 Fact
        
        new_centroid = (old_centroid * (n - 1) + fact_embedding) / n
        
        # 4. 更新 Neo4j 中的 Topic
        self.db_manager.update_topic_centroid(
            topic_id=topic_id,
            new_centroid=new_centroid.tolist(),
            new_fact_count=n
        )
        
        # 5. 更新 Qdrant 中的质心
        self.vector_store.update_topic_centroid(
            topic_id=topic_id,
            project_id=project_id,
            new_centroid=new_centroid
        )
        
        # 6. 触发 LLM 增量更新 Summary（可选：延迟更新）
        # 策略：每累积 5 个新 Fact 才更新一次 Summary
        if n % 5 == 0:
            self._update_topic_summary(
                topic_id=topic_id,
                all_fact_contents=[f['content'] for f in topic_facts['facts']] + [fact_content]
            )
    
    def _generate_topic_summary(
        self,
        fact_contents: List[str],
        is_initial: bool = False,
        previous_summary: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成 Topic 的 Title 和 Summary
        
        Args:
            fact_contents: Fact 内容列表
            is_initial: 是否是初始生成（单 Fact）
            previous_summary: 之前的 Summary（用于增量更新）
        
        Returns:
            {'title': str, 'summary': str}
        """
        if is_initial:
            # 初始 Topic（基于单个 Fact）
            prompt = f"""你是一个知识图谱专家。请为以下事实生成一个简洁的 Topic。

事实: {fact_contents[0]}

请生成：
1. title: Topic 标题（5-10字，概括核心主题）
2. summary: Topic 摘要（50-100字，描述这个事实的含义）

返回 JSON 格式。
"""
        else:
            # 增量更新 Topic
            facts_text = "\n".join([f"{i+1}. {content}" for i, content in enumerate(fact_contents)])
            
            prompt = f"""你是一个知识图谱专家。以下是一组语义高度相似的事实（属于同一个 Topic）。

【当前 Summary】
{previous_summary or "（首次生成）"}

【所有事实】
{facts_text}

请更新 Topic 的 Summary，要求：
1. title: Topic 标题（5-10字，概括核心主题）
2. summary: Topic 摘要（100-200字，整合所有事实的共同主题）

注意：这些事实的语义非常接近（向量距离 < 0.25），它们一定在讨论同一个主题。

返回 JSON 格式。
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个知识总结专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"} if "gpt" in self.llm_model.lower() else None
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'title': result.get('title', 'Untitled Topic'),
                'summary': result.get('summary', '')
            }
        
        except Exception as e:
            print(f"  ⚠️ 生成 Topic Summary 失败: {e}")
            return {
                'title': 'Error Topic',
                'summary': f'Summary 生成失败: {str(e)}'
            }
    
    def _update_topic_summary(
        self,
        topic_id: int,
        all_fact_contents: List[str]
    ):
        """
        更新 Topic 的 Summary（增量更新）
        """
        # 获取当前 Summary
        topic_info = self.db_manager.get_topic_info(topic_id)
        previous_summary = topic_info.get('summary', '')
        
        # 重新生成 Summary
        new_summary = self._generate_topic_summary(
            fact_contents=all_fact_contents,
            is_initial=False,
            previous_summary=previous_summary
        )
        
        # 更新 Neo4j
        self.db_manager.update_topic_summary(
            topic_id=topic_id,
            new_title=new_summary['title'],
            new_summary=new_summary['summary']
        )
        
        # 更新 Qdrant
        self.vector_store.update_topic_metadata(
            topic_id=topic_id,
            title=new_summary['title'],
            summary=new_summary['summary']
        )
        
        print(f"  ✓ Topic {topic_id} Summary 已更新")
    
    def batch_cluster_all_facts(
        self,
        project_id: str,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        批量聚类所有 Facts（用于初始化或重建）
        
        Args:
            project_id: 项目 ID
            force_rebuild: 是否强制重建（删除旧 Topics）
        
        Returns:
            {'total_facts': int, 'total_topics': int, 'topics': [...]}
        """
        print(f"\n[Semantic Clustering] 开始批量聚类 Facts（project_id: {project_id}）")
        
        if force_rebuild:
            print("  - 清理旧 Topics...")
            self.db_manager.delete_all_topics(project_id)
            self.vector_store.delete_topic_centroids(project_id)
        
        # 1. 获取所有 Facts
        all_facts = self.db_manager.get_all_facts(project_id)
        print(f"  - 共 {len(all_facts)} 个 Facts")
        
        if not all_facts:
            return {'total_facts': 0, 'total_topics': 0, 'topics': []}
        
        # 2. 获取所有 Facts 的 embeddings（从 Qdrant）
        fact_embeddings = []
        for fact in all_facts:
            # 从 Qdrant 查询 Fact 的 embedding
            embedding = self.vector_store.get_fact_embedding(fact['fact_id'])
            if embedding is not None:
                fact_embeddings.append(embedding)
            else:
                # 如果没有 embedding，现场生成
                embedding = self.embedder.encode([fact['content']])[0]
                fact_embeddings.append(embedding)
        
        fact_embeddings = np.array(fact_embeddings)
        
        # 3. 遍历每个 Fact，分配到 Topic
        topic_count = 0
        for i, fact in enumerate(all_facts):
            topic_id, is_new = self.assign_fact_to_topic(
                fact_id=fact['fact_id'],
                fact_content=fact['content'],
                fact_embedding=fact_embeddings[i],
                project_id=project_id
            )
            
            if is_new:
                topic_count += 1
        
        # 4. 过滤小 Topics
        print(f"\n  - 过滤 Topic size < {self.min_topic_size}")
        self._filter_small_topics(project_id)
        
        # 5. 获取最终 Topics
        final_topics = self.db_manager.get_all_topics(project_id)
        
        print(f"\n✓ 聚类完成：{len(all_facts)} Facts → {len(final_topics)} Topics")
        
        return {
            'total_facts': len(all_facts),
            'total_topics': len(final_topics),
            'topics': final_topics
        }
    
    def _filter_small_topics(self, project_id: str):
        """
        删除 Fact 数量 < min_topic_size的 Topics
        """
        all_topics = self.db_manager.get_all_topics(project_id)
        
        for topic in all_topics:
            if topic['fact_count'] < self.min_topic_size:
                print(f"  删除小 Topic {topic['topic_id']} (size: {topic['fact_count']})")
                self.db_manager.delete_topic(topic['topic_id'])
                self.vector_store.delete_topic_centroid(topic['topic_id'], project_id)
