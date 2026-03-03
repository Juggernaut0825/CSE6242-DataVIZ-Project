"""
Fact-to-Fact Relation Graph 构建模块
基于实体-社区映射，用 LLM 分析 Facts 之间的逻辑关系
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from datetime import datetime
import asyncio


class EntityCommunityMapper:
    """实体到社区的映射器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化映射器
        
        Args:
            output_dir: GraphRAG 输出目录
        """
        self.output_dir = output_dir
        self.entity_to_communities = {}
    
    def build_mapping(self) -> Dict[str, List[Dict]]:
        """
        从 GraphRAG 输出构建 Entity → Community 映射
        
        Returns:
            {entity_name: [{'community_id': int, 'degree': int}, ...]}
        """
        print("\n正在构建 Entity → Community 映射...")
        
        # 加载数据
        entities_df = pd.read_parquet(self.output_dir / "entities.parquet")
        communities_df = pd.read_parquet(self.output_dir / "communities.parquet")
        
        entity_to_communities = defaultdict(list)
        
        # 遍历每个社区
        for _, comm_row in communities_df.iterrows():
            community_id = comm_row['community']
            
            # 获取该社区的 text_unit_ids
            text_unit_ids = self._parse_list(comm_row.get('text_unit_ids', []))
            if not text_unit_ids:
                continue
            
            text_unit_ids_set = set(str(tu) for tu in text_unit_ids)
            
            # 找到该社区涉及的实体
            for _, entity_row in entities_df.iterrows():
                entity_name = entity_row['title']
                entity_text_units = self._parse_list(entity_row.get('text_unit_ids', []))
                entity_text_units_set = set(str(tu) for tu in entity_text_units)
                
                # 计算该实体在这个社区中的"重要性"（交集数量）
                overlap = len(text_unit_ids_set & entity_text_units_set)
                
                if overlap > 0:
                    entity_to_communities[entity_name].append({
                        'community_id': int(community_id),
                        'degree': overlap,
                        'entity_id': entity_row['id']
                    })
        
        # 对每个实体的社区列表按 degree 排序
        for entity in entity_to_communities:
            entity_to_communities[entity].sort(
                key=lambda x: x['degree'],
                reverse=True
            )
        
        self.entity_to_communities = dict(entity_to_communities)
        
        print(f"  ✓ 构建完成：{len(self.entity_to_communities)} 个实体")
        self._print_statistics()
        
        return self.entity_to_communities
    
    def save_mapping(self, output_path: Path) -> None:
        """保存映射到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.entity_to_communities, f, ensure_ascii=False, indent=2)
        print(f"\n已保存映射: {output_path}")
    
    def load_mapping(self, mapping_path: Path) -> Dict[str, List[Dict]]:
        """从文件加载映射"""
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.entity_to_communities = json.load(f)
        return self.entity_to_communities
    
    @staticmethod
    def _parse_list(val: Any) -> List:
        """解析列表值"""
        import numpy as np
        
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return []
        return []
    
    def _print_statistics(self) -> None:
        """打印统计信息"""
        if not self.entity_to_communities:
            return
        
        # 统计多社区实体
        multi_community_entities = {
            entity: comms 
            for entity, comms in self.entity_to_communities.items() 
            if len(comms) > 1
        }
        
        print(f"  - 单社区实体: {len(self.entity_to_communities) - len(multi_community_entities)}")
        print(f"  - 多社区实体: {len(multi_community_entities)} (桥接实体)")
        
        if multi_community_entities:
            print(f"\n  示例桥接实体:")
            for entity, comms in list(multi_community_entities.items())[:3]:
                comm_ids = [c['community_id'] for c in comms]
                print(f"    - {entity}: 跨社区 {comm_ids}")


class FactCommunityLocator:
    """Fact 到社区的定位器"""
    
    def __init__(self, entity_to_communities: Dict[str, List[Dict]]):
        """
        初始化定位器
        
        Args:
            entity_to_communities: 实体到社区的映射
        """
        self.entity_to_communities = entity_to_communities
    
    def locate(
        self, 
        fact_entities: List[str], 
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        根据 Fact 的实体定位关联的社区
        
        Args:
            fact_entities: Fact 包含的实体列表
            threshold: 最小置信度阈值（相对于最强社区）
        
        Returns:
            [{'community_id': int, 'confidence': float, 'entities': [str]}]
        """
        # 收集所有相关社区及其权重
        community_scores = {}
        entity_community_map = {}
        
        for entity in fact_entities:
            if entity in self.entity_to_communities:
                for comm_info in self.entity_to_communities[entity]:
                    comm_id = comm_info['community_id']
                    degree = comm_info['degree']
                    
                    if comm_id not in community_scores:
                        community_scores[comm_id] = 0
                        entity_community_map[comm_id] = []
                    
                    community_scores[comm_id] += degree
                    entity_community_map[comm_id].append(entity)
        
        # 归一化和过滤
        if not community_scores:
            return []
        
        max_score = max(community_scores.values())
        
        related_communities = []
        for comm_id, score in sorted(
            community_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            confidence = score / max_score
            
            if confidence >= threshold:
                related_communities.append({
                    'community_id': comm_id,
                    'confidence': confidence,
                    'score': score,
                    'entities': entity_community_map[comm_id]
                })
        
        return related_communities


class FactRelationAnalyzer:
    """Fact 关系分析器（使用 LLM）"""
    
    RELATION_TYPES = [
        "Support",
        "Contradict",
        "Cause",
        "Temporal",
        "Elaborate",
        "Analogy",
        "Conditional",
        "Parallel"
    ]
    
    def __init__(
        self, 
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.3
    ):
        """
        初始化分析器
        
        Args:
            api_base: LLM API Base URL
            api_key: LLM API Key
            model: LLM 模型名称
            temperature: 温度参数
        """
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model = model
        self.temperature = temperature
    
    def analyze_batch(
        self,
        new_fact: Dict[str, Any],
        candidate_facts: List[Dict[str, Any]],
        batch_size: int = 15
    ) -> List[Dict[str, Any]]:
        """
        批量分析 Fact 关系
        
        Args:
            new_fact: 新 Fact，包含 fact_id, content, entities
            candidate_facts: 候选 Facts 列表
            batch_size: 每批处理的 Facts 数量
        
        Returns:
            关系列表
        """
        # 按共享实体数量排序（更可能有关系的优先）
        candidate_facts.sort(
            key=lambda f: len(set(f.get('shared_entities', [])) & set(new_fact.get('entities', []))),
            reverse=True
        )
        
        all_relations = []
        
        print(f"\nAnalyzing relationships between Fact {new_fact['fact_id']} and {len(candidate_facts)} candidate Facts...")
        
        for i in range(0, len(candidate_facts), batch_size):
            batch = candidate_facts[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(candidate_facts)-1)//batch_size + 1}...")
            
            try:
                relations = self._analyze_single_batch(new_fact, batch)
                all_relations.extend(relations)
            except Exception as e:
                print(f"  [WARN] Batch analysis failed: {str(e)}")
                continue
        
        print(f"  [OK] Found {len(all_relations)} relationships")
        return all_relations
    
    def _analyze_single_batch(
        self,
        new_fact: Dict[str, Any],
        batch_facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """分析单个批次"""
        
        # Build description of candidate Facts
        candidates_text = ""
        for i, fact in enumerate(batch_facts, 1):
            shared = set(fact.get('shared_entities', [])) & set(new_fact.get('entities', []))
            candidates_text += f"{i}. [ID: {fact['fact_id']}] {fact['content']}\n"
            if shared:
                candidates_text += f"   Shared entities: {', '.join(shared)}\n"
        
        # Construct Prompt (Simplified - No Explanation, No Confidence)
        prompt = f"""Analyze the logical relationships between the new Fact and the following Facts.

[New Fact]
ID: {new_fact['fact_id']}
Content: {new_fact['content']}
Entities: {', '.join(new_fact.get('entities', []))}

[Candidate Facts]
{candidates_text}

[Relation Types]
1. Support: Fact B provides evidence or supplementary information for Fact A
2. Contradict: Fact B has a logical conflict with Fact A
3. Cause: Fact A is the cause of Fact B, or Fact B is the result of Fact A
4. Temporal: Fact A occurs before or after Fact B
5. Elaborate: Fact B elaborates or explains Fact A in detail
6. Analogy: Fact A and Fact B are similar or comparable in some aspects
7. Conditional: If Fact A holds, then Fact B holds
8. Parallel: Fact A and Fact B are parallel options, alternatives, or items at the same level (e.g., "wants A or B", "includes A and B")

[Requirements]
- Only return Facts with clear relationships (ignore irrelevant ones)
- Specify the relationship direction (new_fact → target_fact or target_fact → new_fact)
- If you identify a relationship, just add it - no need to judge confidence

[Output Format] (must be a valid JSON array)
[
  {{
    "target_fact_id": 42,
    "relation_type": "Cause",
    "direction": "new_fact → target_fact"
  }}
]

If no clear relationship exists, return an empty array: []
"""
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional knowledge graph analysis expert skilled at identifying logical relationships between Facts."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"} if "gpt" in self.model.lower() else None
        )
        
        # 解析响应
        content = response.choices[0].message.content
        
        try:
            # 尝试直接解析
            relations = json.loads(content)
            
            # 如果返回的是对象而不是数组，尝试提取
            if isinstance(relations, dict):
                if 'relations' in relations:
                    relations = relations['relations']
                elif 'result' in relations:
                    relations = relations['result']
                else:
                    relations = []
            
            # 添加 new_fact_id
            for rel in relations:
                rel['new_fact_id'] = new_fact['fact_id']
            
            return relations if isinstance(relations, list) else []
        
        except json.JSONDecodeError:
            print(f"  [WARN] JSON parsing failed: {content[:100]}...")
            return []


class FactRelationGraphBuilder:
    """Fact 关系图构建器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化构建器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.graph = {
            "nodes": {},
            "edges": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_nodes": 0,
                "total_edges": 0
            }
        }
    
    def load_graph(self, graph_path: Path) -> Dict[str, Any]:
        """加载现有图"""
        if graph_path.exists():
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.graph = json.load(f)
            print(f"加载现有图: {len(self.graph['nodes'])} 节点, {len(self.graph['edges'])} 边")
        return self.graph
    
    def add_fact_node(self, fact: Dict[str, Any]) -> None:
        """添加 Fact 节点"""
        fact_id = str(fact['fact_id'])
        
        self.graph['nodes'][fact_id] = {
            'fact_id': fact['fact_id'],
            'content': fact['content'],
            'entities': fact.get('entities', []),
            'communities': fact.get('communities', []),
            'created_at': fact.get('created_at', datetime.now().isoformat())
        }
    
    def add_relations(self, relations: List[Dict[str, Any]]) -> None:
        """添加关系边（自动去重）"""
        # 构建现有边的索引（用于去重）
        existing_edges_set = set()
        for edge in self.graph['edges']:
            key = (edge['source'], edge['target'], edge['relation_type'])
            existing_edges_set.add(key)
        
        added_count = 0
        skipped_count = 0
        
        for rel in relations:
            # 解析方向
            if "→" in rel['direction'] or "->" in rel['direction']:
                parts = rel['direction'].replace('→', '->').split('->')
                source_ref = parts[0].strip()
                target_ref = parts[1].strip()
                
                if 'new_fact' in source_ref or source_ref == str(rel['new_fact_id']):
                    source = str(rel['new_fact_id'])
                    target = str(rel['target_fact_id'])
                else:
                    source = str(rel['target_fact_id'])
                    target = str(rel['new_fact_id'])
            else:
                # 默认从 new_fact 指向 target_fact
                source = str(rel['new_fact_id'])
                target = str(rel['target_fact_id'])
            
            # 检查是否已存在相同的边
            key = (source, target, rel['relation_type'])
            
            if key in existing_edges_set:
                # 边已存在，跳过
                skipped_count += 1
            else:
                # 新边，添加
                edge = {
                    "id": f"edge_{len(self.graph['edges'])}",
                    "source": source,
                    "target": target,
                    "relation_type": rel['relation_type'],
                    "created_at": datetime.now().isoformat()
                }
                
                self.graph['edges'].append(edge)
                existing_edges_set.add(key)
                added_count += 1
        
        # 打印统计
        if added_count > 0 or skipped_count > 0:
            print(f"  - 新增 {added_count} 条边, 跳过 {skipped_count} 条重复边")
    
    def save_graph(self, graph_path: Path) -> None:
        """保存图到文件"""
        # 更新元数据
        self.graph['metadata'].update({
            'updated_at': datetime.now().isoformat(),
            'total_nodes': len(self.graph['nodes']),
            'total_edges': len(self.graph['edges'])
        })
        
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)
        
        print(f"\n已保存 Fact Relation Graph: {graph_path}")
        print(f"  - 节点数: {self.graph['metadata']['total_nodes']}")
        print(f"  - 边数: {self.graph['metadata']['total_edges']}")
    
    def get_fact_relations(self, fact_id: int) -> Dict[str, List[Dict]]:
        """获取指定 Fact 的所有关系"""
        fact_id_str = str(fact_id)
        
        outgoing = []
        incoming = []
        
        for edge in self.graph['edges']:
            if edge['source'] == fact_id_str:
                outgoing.append(edge)
            elif edge['target'] == fact_id_str:
                incoming.append(edge)
        
        return {
            'outgoing': outgoing,
            'incoming': incoming,
            'total': len(outgoing) + len(incoming)
        }

