"""
GauzRag 实体映射模块
用于维护 Entity → Facts 的倒排索引，加速 Facts 关系构建
注：此为独立模块，主要实现在 lightrag_graph_builder.py 中
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


class GauzRagEntityMapper:
    """
    基于实体图维护 Entity → Facts 映射
    
    核心优势：
    - O(1) 查找：通过实体快速定位相关 Facts
    - 避免 O(n²)：不需要遍历所有 Facts 对
    - 增量更新：新 Facts 只需更新索引
    """
    
    def __init__(self, output_dir: Path):
        """
        初始化映射器
        
        Args:
            output_dir: 输出目录（用于保存映射文件）
        """
        self.output_dir = output_dir
        self.entity_to_facts: Dict[str, Set[int]] = defaultdict(set)
        self.fact_to_entities: Dict[int, List[str]] = {}
    
    def add_fact(self, fact_id: int, entities: List[str]) -> None:
        """
        添加 Fact 到映射
        
        Args:
            fact_id: Fact ID
            entities: 实体列表（从实体提取器获取）
        """
        # 保存 fact → entities
        self.fact_to_entities[fact_id] = entities
        
        # 更新 entity → facts 倒排索引
        for entity in entities:
            entity_normalized = entity.strip().upper()  # 标准化实体名
            self.entity_to_facts[entity_normalized].add(fact_id)
    
    def get_related_facts(self, entities: List[str]) -> Set[int]:
        """
        通过实体快速查找相关 Facts
        
        Args:
            entities: 实体列表
            
        Returns:
            相关 Fact IDs
        """
        related_facts = set()
        
        for entity in entities:
            entity_normalized = entity.strip().upper()
            related_facts.update(self.entity_to_facts.get(entity_normalized, set()))
        
        return related_facts
    
    def get_fact_entities(self, fact_id: int) -> List[str]:
        """获取 Fact 的实体列表"""
        return self.fact_to_entities.get(fact_id, [])
    
    def get_shared_entities(self, fact_id_1: int, fact_id_2: int) -> List[str]:
        """
        获取两个 Facts 的共享实体
        
        Args:
            fact_id_1: Fact 1 ID
            fact_id_2: Fact 2 ID
            
        Returns:
            共享的实体列表
        """
        entities_1 = set(e.strip().upper() for e in self.fact_to_entities.get(fact_id_1, []))
        entities_2 = set(e.strip().upper() for e in self.fact_to_entities.get(fact_id_2, []))
        
        shared = entities_1 & entities_2
        return list(shared)
    
    def save(self, output_path: Path = None) -> None:
        """
        保存映射到文件
        
        Args:
            output_path: 输出路径（默认为 output_dir/lightrag_entity_to_facts.json）
        """
        if output_path is None:
            output_path = self.output_dir / "lightrag_entity_to_facts.json"
        
        # 转换 set 为 list 以便序列化
        entity_to_facts_serializable = {
            entity: list(facts) 
            for entity, facts in self.entity_to_facts.items()
        }
        
        data = {
            "entity_to_facts": entity_to_facts_serializable,
            "fact_to_entities": self.fact_to_entities,
            "statistics": {
                "total_entities": len(self.entity_to_facts),
                "total_facts": len(self.fact_to_entities),
                "avg_facts_per_entity": sum(len(f) for f in self.entity_to_facts.values()) / max(len(self.entity_to_facts), 1)
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 保存实体映射: {output_path}")
        print(f"    - {data['statistics']['total_entities']} 个实体")
        print(f"    - {data['statistics']['total_facts']} 个 Facts")
        print(f"    - 平均每个实体关联 {data['statistics']['avg_facts_per_entity']:.1f} 个 Facts")
    
    def load(self, input_path: Path = None) -> None:
        """
        从文件加载映射
        
        Args:
            input_path: 输入路径（默认为 output_dir/lightrag_entity_to_facts.json）
        """
        if input_path is None:
            input_path = self.output_dir / "lightrag_entity_to_facts.json"
        
        if not input_path.exists():
            print(f"  ⚠️  映射文件不存在: {input_path}")
            return
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换 list 为 set
        self.entity_to_facts = defaultdict(set, {
            entity: set(facts)
            for entity, facts in data["entity_to_facts"].items()
        })
        
        self.fact_to_entities = {
            int(fact_id): entities 
            for fact_id, entities in data["fact_to_entities"].items()
        }
        
        print(f"  ✓ 加载实体映射: {input_path}")
        if "statistics" in data:
            stats = data["statistics"]
            print(f"    - {stats['total_entities']} 个实体")
            print(f"    - {stats['total_facts']} 个 Facts")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.entity_to_facts:
            return {
                "total_entities": 0,
                "total_facts": 0,
                "avg_facts_per_entity": 0
            }
        
        facts_per_entity = [len(facts) for facts in self.entity_to_facts.values()]
        
        return {
            "total_entities": len(self.entity_to_facts),
            "total_facts": len(self.fact_to_entities),
            "avg_facts_per_entity": sum(facts_per_entity) / len(facts_per_entity),
            "max_facts_per_entity": max(facts_per_entity),
            "min_facts_per_entity": min(facts_per_entity)
        }

