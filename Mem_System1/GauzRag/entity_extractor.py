"""
从 GraphRAG 输出中提取 Fact 的实体
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set


class FactEntityExtractor:
    """从 GraphRAG 输出提取 Fact 对应的实体"""
    
    def __init__(self, output_dir: Path):
        """
        初始化提取器
        
        Args:
            output_dir: GraphRAG 输出目录
        """
        self.output_dir = output_dir
        self.fact_to_entities = {}
    
    def extract_all(self) -> Dict[int, List[str]]:
        """
        提取所有 Facts 的实体（支持批量模式）
        
        Returns:
            {fact_id: [entity_name, ...]}
        """
        print("\n正在从 GraphRAG 输出提取 Fact 实体...")
        
        # 加载数据
        documents_df = pd.read_parquet(self.output_dir / "documents.parquet")
        text_units_df = pd.read_parquet(self.output_dir / "text_units.parquet")
        entities_df = pd.read_parquet(self.output_dir / "entities.parquet")
        
        # 检查是否存在批量文档
        has_batch_docs = False
        for _, row in documents_df.iterrows():
            meta = row.get("metadata")
            try:
                if isinstance(meta, dict):
                    doc_type = meta.get("type")
                elif isinstance(meta, str):
                    doc_type = json.loads(meta).get("type")
                else:
                    doc_type = None
                
                if doc_type == "batch":
                    has_batch_docs = True
                    break
            except Exception:
                pass
        
        # 如果存在批量文档，使用批量提取逻辑
        if has_batch_docs:
            print("  → 检测到批量模式，使用 XML 标记解析...")
            from .smart_batch_builder import map_entities_to_facts
            self.fact_to_entities = map_entities_to_facts(entities_df, text_units_df, documents_df)
            print(f"  ✓ 批量提取完成：{len(self.fact_to_entities)} 个 Facts")
            self._print_statistics()
            return self.fact_to_entities
        
        # 否则使用独立模式逻辑
        print("  → 使用独立模式解析...")
        
        # 1. 构建 document_id -> fact_id 映射
        doc_to_fact = {}
        for _, row in documents_df.iterrows():
            doc_id = str(row["id"])
            meta = row.get("metadata")
            
            fact_id = None
            try:
                if isinstance(meta, dict):
                    fact_id = meta.get("fact_id")
                elif isinstance(meta, str):
                    fact_id = json.loads(meta).get("fact_id")
            except Exception:
                pass
            
            if fact_id is not None:
                doc_to_fact[doc_id] = int(fact_id)
        
        # 2. 构建 text_unit_id -> document_id 映射
        tu_to_doc = {}
        doc_id_col = self._find_column(text_units_df, ["document_ids", "document_id", "documents"])
        
        if doc_id_col:
            for _, row in text_units_df.iterrows():
                tu_id = str(row["id"])
                doc_ids = self._parse_list(row.get(doc_id_col))
                if doc_ids:
                    tu_to_doc[tu_id] = str(doc_ids[0])
        
        # 3. 构建 fact_id -> entities 映射
        fact_to_entities = {}
        
        for _, entity_row in entities_df.iterrows():
            entity_name = entity_row['title']
            text_unit_ids = self._parse_list(entity_row.get('text_unit_ids', []))
            
            for tu_id in text_unit_ids:
                tu_id_str = str(tu_id)
                if tu_id_str in tu_to_doc:
                    doc_id = tu_to_doc[tu_id_str]
                    if doc_id in doc_to_fact:
                        fact_id = doc_to_fact[doc_id]
                        
                        if fact_id not in fact_to_entities:
                            fact_to_entities[fact_id] = set()
                        
                        fact_to_entities[fact_id].add(entity_name)
        
        # 转换 set 为 list
        self.fact_to_entities = {
            fact_id: list(entities)
            for fact_id, entities in fact_to_entities.items()
        }
        
        print(f"  ✓ 提取完成：{len(self.fact_to_entities)} 个 Facts")
        self._print_statistics()
        
        return self.fact_to_entities
    
    def get_entities(self, fact_id: int) -> List[str]:
        """获取指定 Fact 的实体"""
        return self.fact_to_entities.get(fact_id, [])
    
    def save(self, output_path: Path) -> None:
        """保存到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.fact_to_entities, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 Fact 实体映射: {output_path}")
    
    def load(self, mapping_path: Path) -> Dict[int, List[str]]:
        """从文件加载"""
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 确保 key 是 int
            self.fact_to_entities = {int(k): v for k, v in data.items()}
        return self.fact_to_entities
    
    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
        """查找列名"""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    @staticmethod
    def _parse_list(val) -> List:
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
        if not self.fact_to_entities:
            return
        
        entity_counts = [len(entities) for entities in self.fact_to_entities.values()]
        avg_entities = sum(entity_counts) / len(entity_counts)
        
        facts_without_entities = sum(1 for count in entity_counts if count == 0)
        
        print(f"  - 平均每个 Fact 有 {avg_entities:.1f} 个实体")
        print(f"  - 没有实体的 Facts: {facts_without_entities}")
        
        if entity_counts:
            print(f"  - 最多实体的 Fact: {max(entity_counts)} 个")

