"""
Community 搜索召回模块
基于 Community Report 的 embedding 进行语义召回
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from .embedder import DashScopeEmbedder

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None


class CommunitySearcher:
    """Community 召回器"""
    
    def __init__(self, embeddings_path: Path, community_facts_path: Path):
        """
        初始化召回器
        
        Args:
            embeddings_path: community_embeddings.pkl 文件路径
            community_facts_path: community_facts.json 文件路径
        """
        print("正在加载 Community 索引...")
        
        # 加载 embeddings
        with open(embeddings_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
        
        # 加载完整的 community facts 数据
        with open(community_facts_path, 'r', encoding='utf-8') as f:
            self.community_facts = json.load(f)
        
        # 加载 embedding 模型
        if cosine_similarity is None:
            raise RuntimeError(
                "未安装 scikit-learn。请安装: pip install scikit-learn"
            )
        
        model_name = self.embedding_data['model_name']
        print(f"正在加载模型: {model_name}")
        self.model = DashScopeEmbedder()
        
        print("索引加载完成")
        print(f"   - 社区数量: {len(self.embedding_data['communities'])}")
        print(f"   - Embedding 维度: {self.embedding_data['embedding_dim']}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        return_facts: bool = True,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        搜索相关的 communities
        
        Args:
            query: 用户查询
            top_k: 返回前 k 个最相关的社区
            return_facts: 是否返回社区内的 facts
            min_score: 最低相似度分数阈值
        
        Returns:
            搜索结果列表
        """
        print(f"\n查询: {query}")
        
        # 1. 对查询生成 embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 2. 计算与所有 community 的相似度
        similarities = cosine_similarity(
            query_embedding,
            self.embedding_data['embeddings']
        )[0]
        
        # 3. 排序并选出 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. 构建返回结果
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            # 过滤低分结果
            if score < min_score:
                continue
            
            comm_data = self.embedding_data['communities'][idx]
            comm_key = comm_data['community_key']
            comm_full_data = self.community_facts[comm_key]
            
            result = {
                'community_id': comm_data['community_id'],
                'community_name': comm_data['report'].get('title', f"Community {comm_data['community_id']}"),
                'relevance_score': score,
                'report': comm_data['report'],
                'fact_count': comm_data['fact_count']
            }
            
            # 添加 facts（如果需要）
            if return_facts:
                result['facts'] = comm_full_data['facts']
            
            results.append(result)
            
            print(f"  - Community {result['community_id']}: {result['community_name'][:50]}... (相似度: {score:.3f})")
        
        return results
    
    def search_with_deduplication(
        self,
        query: str,
        top_k: int = 5,
        dedupe_threshold: float = 0.85,
        max_facts_per_community: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索并对 facts 进行语义去重
        
        Args:
            query: 用户查询
            top_k: 返回前 k 个最相关的社区
            dedupe_threshold: 语义去重的相似度阈值
            max_facts_per_community: 每个社区最多返回的 facts 数量
        
        Returns:
            去重后的结果
        """
        # 1. 先召回相关社区
        results = self.search(query, top_k=top_k, return_facts=True)
        
        # 2. 对每个社区内的 facts 进行去重
        print(f"\n正在对 facts 进行语义去重（阈值: {dedupe_threshold}）...")
        
        for result in results:
            facts = result['facts']
            
            if len(facts) <= 1:
                continue
            
            # 生成 facts 的 embeddings
            fact_texts = [f['content'] for f in facts]
            fact_embeddings = self.model.encode(fact_texts, convert_to_numpy=True)
            
            # 去重
            deduplicated_facts = []
            seen_embeddings = []
            
            for i, fact in enumerate(facts):
                is_duplicate = False
                
                for seen_emb in seen_embeddings:
                    similarity = cosine_similarity(
                        [fact_embeddings[i]],
                        [seen_emb]
                    )[0][0]
                    
                    if similarity > dedupe_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    deduplicated_facts.append(fact)
                    seen_embeddings.append(fact_embeddings[i])
                    
                    # 限制数量
                    if len(deduplicated_facts) >= max_facts_per_community:
                        break
            
            # 更新结果
            original_count = len(facts)
            result['facts'] = deduplicated_facts
            result['fact_count'] = len(deduplicated_facts)
            result['original_fact_count'] = original_count
            
            if original_count > len(deduplicated_facts):
                print(f"  Community {result['community_id']}: {original_count} → {len(deduplicated_facts)} facts")
        
        return results


class EmbeddingIndexBuilder:
    """Embedding 索引构建器"""
    
    @staticmethod
    def build(
        community_facts_path: Path,
        output_path: Path,
        embedder: DashScopeEmbedder
    ) -> None:
        """
        为所有 community reports 生成 embedding 索引
        
        Args:
            community_facts_path: community_facts.json 文件路径
            output_path: 输出的 embedding 文件路径
            embedder: Embedding 模型
        """
        print(f"\n正在读取 community facts: {community_facts_path}")
        with open(community_facts_path, 'r', encoding='utf-8') as f:
            community_facts = json.load(f)
        
        print(f"找到 {len(community_facts)} 个社区\n")
        
        # 准备数据
        communities_data = []
        texts_to_embed = []
        
        for comm_key, comm_data in sorted(community_facts.items()):
            comm_id = comm_data['community_id']
            report = comm_data.get('report', {})
            
            # 构造用于 embedding 的文本（标题 + 摘要）
            if report and report.get('title') and report.get('summary'):
                text = f"{report['title']}. {report['summary']}"
            elif report and report.get('title'):
                text = report['title']
            elif report and report.get('summary'):
                text = report['summary']
            else:
                # 降级方案：使用前5条 facts 的拼接
                facts_text = " ".join([f['content'] for f in comm_data['facts'][:5]])
                text = f"Community {comm_id}. {facts_text}"
            
            communities_data.append({
                'community_id': comm_id,
                'community_key': comm_key,
                'text': text,
                'report': report,
                'fact_count': comm_data['fact_count'],
                'fact_ids': [f['fact_id'] for f in comm_data['facts']]
            })
            
            texts_to_embed.append(text)
            
            # 显示预览
            print(f"Community {comm_id}: {text[:100]}...")
        
        if len(texts_to_embed) == 0:
            print("\n警告: 没有社区数据，无法构建 Embedding 索引")
            print("可能原因:")
            print("  1. Facts 数量太少（建议至少 50 条以上）")
            print("  2. GraphRAG 社区检测失败")
            print("  3. Community 映射提取失败")
            return
        
        print(f"\n正在生成 {len(texts_to_embed)} 个 embeddings...")
        embeddings = embedder.encode(
            texts_to_embed,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 保存结果
        embedding_data = {
            'model_name': embedder.model,
            'communities': communities_data,
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1]
        }
        
        print(f"\n正在保存到: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"\nEmbedding 索引构建完成!")
        print(f"   - 社区数量: {len(communities_data)}")
        print(f"   - Embedding 维度: {embeddings.shape[1]}")
        print(f"   - 模型: {embedder.model}")
        print(f"   - 文件大小: {output_path.stat().st_size / 1024:.2f} KB")


class FactSearcher:
    """Fact 级别的搜索召回器"""
    
    def __init__(
        self, 
        fact_embeddings_path: Path, 
        community_facts_path: Path,
        fact_community_reports_path: Optional[Path] = None
    ):
        """
        初始化 Fact 搜索器
        
        Args:
            fact_embeddings_path: fact_embeddings.pkl 文件路径
            community_facts_path: community_facts.json 文件路径（GraphRAG Entity Communities）
            fact_community_reports_path: fact_community_reports.json 文件路径（Fact Communities/Topics）
        """
        print("正在加载 Fact 索引...")
        
        # 加载 fact embeddings
        with open(fact_embeddings_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
        
        # 加载完整的 community facts 数据（GraphRAG Entity Communities）
        with open(community_facts_path, 'r', encoding='utf-8') as f:
            self.community_facts = json.load(f)
        
        # 构建 fact_id -> entity community 的映射
        self.fact_to_entity_community = {}
        for comm_key, comm_data in self.community_facts.items():
            for fact in comm_data['facts']:
                self.fact_to_entity_community[fact['fact_id']] = {
                    'community_id': comm_data['community_id'],
                    'community_key': comm_key,
                    'community_type': 'entity',  # GraphRAG Entity Community
                    'report': comm_data.get('report', {})
                }
        
        # 加载 Fact Community Reports (Topics)
        self.fact_community_reports = None
        self.fact_to_topic = {}
        
        if fact_community_reports_path and fact_community_reports_path.exists():
            print("正在加载 Fact Community (Topic) 信息...")
            with open(fact_community_reports_path, 'r', encoding='utf-8') as f:
                self.fact_community_reports = json.load(f)
            
            # 构建 fact_id -> topic 的映射
            for topic_id, topic_data in self.fact_community_reports.items():
                for fact in topic_data['facts']:
                    self.fact_to_topic[fact['fact_id']] = {
                        'topic_id': topic_id,
                        'topic_type': 'fact_community',  # Fact Community (基于逻辑关系)
                        'report': topic_data['report'],
                        'statistics': topic_data.get('statistics', {})
                    }
            
            print(f"   - Fact Topics 数量: {len(self.fact_community_reports)}")
        else:
            print("   - 未找到 Fact Community Reports，将使用 Entity Community")
        
        # 加载 embedding 模型
        if cosine_similarity is None:
            raise RuntimeError(
                "未安装 scikit-learn。请安装: pip install scikit-learn"
            )
        
        model_name = self.embedding_data['model_name']
        print(f"正在加载模型: {model_name}")
        self.model = DashScopeEmbedder()
        
        print("索引加载完成")
        print(f"   - Facts 数量: {len(self.embedding_data['facts'])}")
        print(f"   - Embedding 维度: {self.embedding_data['embedding_dim']}")
        
        # 检查是否有对话原文的 embeddings
        if 'conversation_embeddings' in self.embedding_data:
            print(f"   - 对话原文 Embeddings: 已加载")
        else:
            print(f"   - 对话原文 Embeddings: 未找到（旧版索引）")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        include_community: bool = True,
        search_mode: str = "fact"
    ) -> List[Dict[str, Any]]:
        """
        搜索相关的 facts（支持三级召回）
        
        Args:
            query: 用户查询
            top_k: 返回前 k 个最相关的 facts
            min_score: 最低相似度分数阈值
            include_community: 是否包含所属 community 的 report
            search_mode: 搜索模式 - "fact" (Fact内容), "conversation" (对话原文), "hybrid" (混合)
        
        Returns:
            搜索结果列表，每个结果包含:
            - fact_id: Fact ID
            - content: Fact 内容
            - conversation_text: 对话原文
            - relevance_score: 相似度分数
            - topic_id/community_id: 所属 Topic/Community ID (如果 include_community=True)
            - topic_name/community_name: 所属 Topic/Community 名称 (如果 include_community=True)
            - topic_report/community_report: 所属 Topic/Community 报告 (如果 include_community=True)
        """
        print(f"\n查询 Facts: {query} [模式: {search_mode}]")
        
        # 1. 对查询生成 embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 2. 根据搜索模式计算相似度
        if search_mode == "conversation" and 'conversation_embeddings' in self.embedding_data:
            # 使用对话原文的 embedding
            similarities = cosine_similarity(
                query_embedding,
                self.embedding_data['conversation_embeddings']
            )[0]
        elif search_mode == "hybrid" and 'conversation_embeddings' in self.embedding_data:
            # 混合模式：同时考虑 fact 和对话原文的相似度
            fact_sims = cosine_similarity(
                query_embedding,
                self.embedding_data['embeddings']
            )[0]
            conv_sims = cosine_similarity(
                query_embedding,
                self.embedding_data['conversation_embeddings']
            )[0]
            # 加权平均（可以调整权重）
            similarities = 0.6 * fact_sims + 0.4 * conv_sims
        else:
            # 默认使用 fact 内容的 embedding
            similarities = cosine_similarity(
                query_embedding,
                self.embedding_data['embeddings']
            )[0]
        
        # 3. 排序并选出 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. 构建返回结果（三级召回：对话原文 + Facts + Topic）
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            # 过滤低分结果
            if score < min_score:
                continue
            
            fact_data = self.embedding_data['facts'][idx]
            fact_id = fact_data['fact_id']
            
            result = {
                'fact_id': fact_id,
                'content': fact_data['content'],
                'conversation_text': fact_data.get('conversation_text', ''),  # 第1级：对话原文
                'relevance_score': score
            }
            
            # 第3级：添加所属社区/topic 信息
            # 优先返回 Fact Community (Topic)，如果不存在则降级到 Entity Community
            if include_community:
                if fact_id in self.fact_to_topic:
                    # 返回 Fact Community (Topic) 信息
                    topic_info = self.fact_to_topic[fact_id]
                    result['topic_id'] = topic_info['topic_id']
                    result['topic_name'] = topic_info['report'].get('title', topic_info['topic_id'])
                    result['topic_summary'] = topic_info['report'].get('summary', '')
                    result['topic_report'] = topic_info['report']
                    result['topic_statistics'] = topic_info.get('statistics', {})
                    result['community_type'] = 'topic'  # Fact Community
                    
                elif fact_id in self.fact_to_entity_community:
                    # Fallback: 返回 GraphRAG Entity Community 信息
                    comm_info = self.fact_to_entity_community[fact_id]
                    result['community_id'] = comm_info['community_id']
                    result['community_name'] = comm_info['report'].get(
                        'title', 
                        f"Community {comm_info['community_id']}"
                    )
                    result['community_report'] = comm_info['report']
                    result['community_type'] = 'entity'  # GraphRAG Entity Community
                    result['_note'] = 'Fact Community 不可用，返回 Entity Community'
            
            results.append(result)
            
            preview = result['content'][:80].replace('\n', ' ')
            comm_info_str = f"[{result.get('community_type', 'N/A')}]" if include_community else ""
            print(f"  - Fact {fact_id} {comm_info_str}: {preview}... (相似度: {score:.3f})")
        
        return results


# FactEmbeddingIndexBuilder 已移除
# 请使用 GauzRag/vector_store.py 中的 ChromaDB 方案