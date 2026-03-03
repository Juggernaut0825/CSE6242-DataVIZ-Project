"""
Fact Community Detector
对 Fact-Fact 关系图进行社区检测
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import networkx as nx
from networkx.algorithms import community
from openai import OpenAI


class FactCommunityDetector:
    """基于 Fact-Fact 关系图的社区检测器"""
    
    def __init__(self, graph_path: Path):
        """
        初始化检测器
        
        Args:
            graph_path: fact_relations.json 路径
        """
        self.graph_path = graph_path
        self.graph_data = None
        self.nx_graph = None
        self.fact_communities = None
    
    def load_graph(self) -> Dict[str, Any]:
        """加载 Fact 关系图"""
        with open(self.graph_path, 'r', encoding='utf-8') as f:
            self.graph_data = json.load(f)
        
        print(f"加载 Fact 关系图: {len(self.graph_data['nodes'])} 节点, {len(self.graph_data['edges'])} 边")
        return self.graph_data
    
    def build_networkx_graph(self) -> nx.Graph:
        """构建 NetworkX 图用于聚类"""
        # 创建有向图（可以考虑关系方向）
        self.nx_graph = nx.DiGraph()
        
        # 添加节点
        for node_id, node_data in self.graph_data['nodes'].items():
            self.nx_graph.add_node(
                node_id,
                fact_id=node_data['fact_id'],
                content=node_data['content'],
                entities=node_data.get('entities', [])
            )
        
        # 添加边（使用置信度作为权重）
        skipped_edges = 0
        total_edges_in_json = len(self.graph_data['edges'])
        
        for edge in self.graph_data['edges']:
            # 只添加两端节点都存在的边
            if edge['source'] in self.graph_data['nodes'] and edge['target'] in self.graph_data['nodes']:
                self.nx_graph.add_edge(
                    edge['source'],
                    edge['target'],
                    weight=edge['confidence'],
                    relation_type=edge['relation_type']
                )
            else:
                skipped_edges += 1
        
        valid_edges = total_edges_in_json - skipped_edges
        actual_edges = self.nx_graph.number_of_edges()
        duplicate_edges = valid_edges - actual_edges
        
        if skipped_edges > 0:
            print(f"  ⚠️  跳过了 {skipped_edges} 条无效边（节点不存在）")
        if duplicate_edges > 0:
            print(f"  ⚠️  去重了 {duplicate_edges} 条重复边（相同的 source→target）")
        
        print(f"构建 NetworkX 图: {self.nx_graph.number_of_nodes()} 节点, {actual_edges} 边 (JSON中 {total_edges_in_json} 条)")
        return self.nx_graph
    
    def detect_connected_components(
        self,
        min_size: int = 5
    ) -> Dict[str, List[str]]:
        """
        检测连通分量（替代复杂的社区检测）
        
        连通分量 = 一组通过语义关系连接的 Facts
        只保留 size >= min_size 的连通分量作为 Topics
        
        Args:
            min_size: 最小 Facts 数量（默认 5）
        
        Returns:
            {topic_id: [fact_ids]}
        """
        if self.nx_graph is None:
            self.build_networkx_graph()
        
        # 转为无向图（因为语义关系可能是双向的）
        undirected_graph = self.nx_graph.to_undirected()
        
        print(f"\n检测连通分量...")
        
        # 获取所有连通分量
        connected_components = list(nx.connected_components(undirected_graph))
        
        # 筛选 size >= min_size 的连通分量
        valid_components = [
            comp for comp in connected_components 
            if len(comp) >= min_size
        ]
        
        # 转换为字典格式
        self.fact_communities = {}
        for i, comp_nodes in enumerate(valid_components):
            self.fact_communities[f"fact_topic_{i}"] = list(comp_nodes)
        
        total_count = len(connected_components)
        valid_count = len(valid_components)
        ignored_count = total_count - valid_count
        
        print(f"检测到 {total_count} 个连通分量")
        print(f"  - 有效 Topics（>= {min_size} Facts）: {valid_count}")
        print(f"  - 小分量（< {min_size} Facts）: {ignored_count}（已忽略）")
        
        # 打印详细统计
        for topic_id, fact_ids in self.fact_communities.items():
            print(f"  {topic_id}: {len(fact_ids)} facts")
        
        return self.fact_communities
    
    def calculate_modularity(self) -> float:
        """计算模块度（衡量社区质量）"""
        if self.fact_communities is None or self.nx_graph is None:
            raise ValueError("请先运行 detect_communities()")
        
        # 转为无向图
        undirected_graph = self.nx_graph.to_undirected()
        
        # 转换社区格式
        partition = {}
        for comm_id, fact_ids in self.fact_communities.items():
            for fact_id in fact_ids:
                partition[fact_id] = comm_id
        
        # 计算模块度
        mod = community.modularity(
            undirected_graph,
            [set(fact_ids) for fact_ids in self.fact_communities.values()],
            weight='weight'
        )
        
        print(f"\n模块度: {mod:.4f} (越接近1越好)")
        return mod
    
    def analyze_communities(self) -> Dict[str, Any]:
        """分析每个社区的特征"""
        if self.fact_communities is None:
            raise ValueError("请先运行 detect_communities()")
        
        analysis = {}
        
        for comm_id, fact_ids in self.fact_communities.items():
            # 获取社区内的 Facts
            facts = []
            all_entities = []
            skipped_facts = []
            
            for fact_id in fact_ids:
                # 容错：跳过不存在的节点
                if fact_id not in self.graph_data['nodes']:
                    skipped_facts.append(fact_id)
                    continue
                
                node_data = self.graph_data['nodes'][fact_id]
                facts.append({
                    'fact_id': node_data['fact_id'],
                    'content': node_data['content']
                })
                all_entities.extend(node_data.get('entities', []))
            
            # 如果有跳过的节点，打印警告
            if skipped_facts:
                print(f"  ⚠️  社区 {comm_id} 跳过了 {len(skipped_facts)} 个不存在的节点: {skipped_facts[:5]}")
            
            # 如果社区为空，跳过
            if not facts:
                print(f"  ⚠️  社区 {comm_id} 没有有效的 Facts，跳过")
                continue
            
            # 统计实体频率
            entity_freq = {}
            for entity in all_entities:
                entity_freq[entity] = entity_freq.get(entity, 0) + 1
            
            # 计算社区内边数和跨社区边数
            internal_edges = 0
            external_edges = 0
            relation_types = {}
            
            for edge in self.graph_data['edges']:
                source_in = edge['source'] in fact_ids
                target_in = edge['target'] in fact_ids
                
                if source_in and target_in:
                    internal_edges += 1
                    rel_type = edge['relation_type']
                    relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
                elif source_in or target_in:
                    external_edges += 1
            
            # 计算密度
            max_edges = len(fact_ids) * (len(fact_ids) - 1)
            density = internal_edges / max_edges if max_edges > 0 else 0
            
            analysis[comm_id] = {
                'fact_count': len(facts),
                'facts': facts,
                'top_entities': sorted(
                    entity_freq.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'internal_edges': internal_edges,
                'external_edges': external_edges,
                'density': density,
                'relation_types': relation_types
            }
        
        return analysis
    
    def generate_community_reports(
        self,
        analysis: Dict[str, Any],
        llm_client: OpenAI,
        model: str,
        output_path: Path
    ) -> Dict[str, Dict[str, Any]]:
        """
        为每个 Fact 社区生成报告（类似 GraphRAG 的 Community Report）
        
        Args:
            analysis: analyze_communities() 的返回值
            llm_client: OpenAI 客户端
            model: LLM 模型名称
            output_path: 输出路径
        
        Returns:
            {community_id: report}
        """
        reports = {}
        
        print("\n生成 Fact 社区报告...")
        
        for comm_id, comm_data in analysis.items():
            print(f"  生成 {comm_id} 的报告...")
            
            # 构建 Prompt
            facts_text = "\n".join([
                f"{i+1}. {fact['content']}"
                for i, fact in enumerate(comm_data['facts'])
            ])
            
            entities_text = ", ".join([
                f"{entity}({count})"
                for entity, count in comm_data['top_entities']
            ])
            
            relation_types_text = ", ".join([
                f"{rel_type}({count})"
                for rel_type, count in comm_data['relation_types'].items()
            ])
            
            prompt = f"""你是一个知识图谱分析专家。请为以下连通的 Facts 生成一个 Topic Summary。

【连通图信息】
- Facts 数量: {comm_data['fact_count']}
- 主要实体: {entities_text}
- 关系类型: {relation_types_text}

【Facts 列表（这些 Facts 通过语义关系互相连接）】
{facts_text}

请生成简洁的 Topic Summary:
1. title: Topic 标题（10字以内，概括主题）
2. summary: Topic 摘要（100-200字，描述这些 Facts 的整体含义和关联）
3. key_points: 3-5 个关键点（每个一句话）
4. importance: 重要性评分（0-10）

返回 JSON 格式。
"""
            
            try:
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个知识总结专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"} if "gpt" in model.lower() else None
                )
                
                report_text = response.choices[0].message.content
                report = json.loads(report_text)
                
                # 简化结构
                reports[comm_id] = {
                    'topic_id': comm_id,
                    'fact_count': comm_data['fact_count'],
                    'fact_ids': [f['fact_id'] for f in comm_data['facts']],
                    'title': report.get('title', ''),
                    'summary': report.get('summary', ''),
                    'key_points': report.get('key_points', []),
                    'importance': report.get('importance', 5.0),
                    'top_entities': comm_data['top_entities'][:5]
                }
                
            except Exception as e:
                print(f"    ⚠️  生成报告失败: {str(e)}")
                reports[comm_id] = {
                    'topic_id': comm_id,
                    'fact_count': comm_data['fact_count'],
                    'fact_ids': [f['fact_id'] for f in comm_data['facts']],
                    'title': f"Topic {comm_id}",
                    'summary': f"Topic Summary 生成失败: {str(e)}",
                    'key_points': [],
                    'importance': 0,
                    'top_entities': comm_data['top_entities'][:5]
                }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Fact 社区报告已保存到: {output_path}")
        return reports
    
    def compare_with_graphrag(
        self,
        graphrag_communities_path: Path
    ) -> Dict[str, Any]:
        """
        对比 Fact Communities 和 GraphRAG Communities
        
        Args:
            graphrag_communities_path: community_facts.json 路径
        
        Returns:
            对比分析结果
        """
        if self.fact_communities is None:
            raise ValueError("请先运行 detect_communities()")
        
        # 加载 GraphRAG Communities
        with open(graphrag_communities_path, 'r', encoding='utf-8') as f:
            graphrag_comms = json.load(f)
        
        print("\n对比 Fact Communities vs GraphRAG Communities:")
        print(f"  Fact Communities: {len(self.fact_communities)}")
        print(f"  GraphRAG Communities: {len(graphrag_comms)}")
        
        # 计算重叠度
        comparison = {}
        
        for fact_comm_id, fact_ids in self.fact_communities.items():
            fact_ids_set = set(int(fid) for fid in fact_ids)
            
            best_match = None
            best_overlap = 0
            
            for graphrag_comm_key, graphrag_comm_data in graphrag_comms.items():
                graphrag_fact_ids = set(f['fact_id'] for f in graphrag_comm_data['facts'])
                
                overlap = len(fact_ids_set & graphrag_fact_ids)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = graphrag_comm_key
            
            overlap_ratio = best_overlap / len(fact_ids_set) if fact_ids_set else 0
            
            comparison[fact_comm_id] = {
                'fact_count': len(fact_ids_set),
                'best_match_graphrag': best_match,
                'overlap_count': best_overlap,
                'overlap_ratio': overlap_ratio
            }
            
            print(f"\n  {fact_comm_id}:")
            print(f"    Facts: {len(fact_ids_set)}")
            print(f"    最匹配: {best_match} (重叠 {best_overlap}/{len(fact_ids_set)} = {overlap_ratio:.1%})")
        
        return comparison

