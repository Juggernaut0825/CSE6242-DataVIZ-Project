"""
Community-Facts 映射模块
将图谱中的 community 映射回原始的 facts
"""
import json
from pathlib import Path
from typing import Dict, Set, List, Any
from collections import defaultdict
import pandas as pd


class CommunityMapper:
    """Community-Facts 映射器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化映射器
        
        Args:
            output_dir: GraphRAG 输出目录
        """
        self.output_dir = output_dir
    
    def build_mapping(
        self,
        fact_contents: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        构建 community 到 facts 的映射
        
        Args:
            fact_contents: {fact_id: content} 字典
        
        Returns:
            映射结果字典
        """
        print("正在从 parquet 文件中提取 community-fact_id 映射...")
        
        # 提取映射和 reports
        community_facts, community_reports = self._extract_from_parquet()
        
        print(f"发现 {len(community_facts)} 个社区")
        
        # 收集所有需要查询的 fact_ids
        all_fact_ids = set()
        for fact_ids in community_facts.values():
            all_fact_ids.update(fact_ids)
        
        print(f"涉及 {len(all_fact_ids)} 个唯一的 fact_id")
        print(f"提供了 {len(fact_contents)} 条 fact 内容")
        
        # 检查是否有遗漏的 facts（未被任何社区包含）
        all_provided_fact_ids = set(fact_contents.keys())
        missing_fact_ids = all_provided_fact_ids - all_fact_ids
        
        if missing_fact_ids:
            print(f"\n⚠️  警告: {len(missing_fact_ids)} 条 facts 未被任何社区包含!")
            print(f"   - 已分类: {len(all_fact_ids)} 条")
            print(f"   - 未分类: {len(missing_fact_ids)} 条")
            print(f"   - 覆盖率: {len(all_fact_ids) / len(all_provided_fact_ids) * 100:.1f}%")
            print(f"\n正在创建 'uncategorized' 社区收集未分类的 facts...")
            
            # 添加未分类社区
            community_facts[-1] = missing_fact_ids  # 使用 -1 作为特殊社区 ID
            all_fact_ids.update(missing_fact_ids)
        
        # 构建最终结果
        result = {}
        
        def sort_key(item):
            comm = str(item[0])
            return int(comm) if comm.isdigit() else 999
        
        for comm, fact_ids in sorted(community_facts.items(), key=sort_key):
            facts = []
            for fid in sorted(fact_ids):
                content = fact_contents.get(fid, f"[Fact {fid} not found]")
                facts.append({
                    "fact_id": fid,
                    "content": content
                })
            
            # 获取该社区的 report
            report_data = {}
            if comm == -1:
                # 未分类社区的特殊 report
                report_data = {
                    "title": "Uncategorized Facts",
                    "summary": f"这个特殊社区包含了 {len(facts)} 条未被 GraphRAG 社区检测覆盖的 facts。这些 facts 可能是：1) 实体关系较弱；2) 信息较为独立；3) 与其他 facts 的连接不够紧密。",
                    "rating": 5.0,
                    "rating_explanation": "未分类社区，包含独立或弱关联的 facts"
                }
            elif community_reports is not None:
                comm_report = community_reports[community_reports['community'] == comm]
                if not comm_report.empty:
                    report_row = comm_report.iloc[0]
                    report_data = {
                        "title": str(report_row.get('title', f'Community {comm}')),
                        "summary": str(report_row.get('summary', '')),
                        "rating": float(report_row.get('rank', 0.0)) if pd.notna(report_row.get('rank')) else None,
                        "rating_explanation": str(report_row.get('rating_explanation', ''))
                    }
            
            result[f"community_{comm}"] = {
                "community_id": comm,
                "fact_count": len(facts),
                "facts": facts,
                "report": report_data
            }
        
        return result
    
    def save_mapping(
        self,
        mapping: Dict[str, Any],
        json_path: Path,
        md_path: Path
    ) -> None:
        """
        保存映射结果
        
        Args:
            mapping: 映射结果
            json_path: JSON 输出路径
            md_path: Markdown 输出路径
        """
        # 保存 JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"\n已生成: {json_path.resolve()}")
        
        # 保存 Markdown
        self._save_markdown(mapping, md_path)
        print(f"已生成: {md_path.resolve()}")
        
        # 打印统计
        self._print_statistics(mapping)
    
    def _extract_from_parquet(self) -> tuple:
        """从 parquet 文件中提取映射和 reports"""
        communities = pd.read_parquet(self.output_dir / "communities.parquet")
        text_units = pd.read_parquet(self.output_dir / "text_units.parquet")
        documents = pd.read_parquet(self.output_dir / "documents.parquet")
        
        # 加载 community reports
        community_reports_path = self.output_dir / "community_reports.parquet"
        community_reports = None
        if community_reports_path.exists():
            community_reports = pd.read_parquet(community_reports_path)
        
        # 1. 构建 document_id -> fact_id 映射
        doc_to_fact = self._build_doc_to_fact_map(documents)
        
        # 2. 构建 text_unit_id -> document_id 映射
        tu_to_doc = self._build_tu_to_doc_map(text_units)
        
        # 3. 遍历 communities，通过 text_unit_ids 找到 fact_ids
        community_facts = self._build_community_facts_map(communities, tu_to_doc, doc_to_fact)
        
        return community_facts, community_reports
    
    def _build_doc_to_fact_map(self, documents: pd.DataFrame) -> Dict[str, Any]:
        """
        构建 document_id -> fact_id(s) 映射
        
        Returns:
            {
                "doc_id": fact_id  # 单fact模式
                或
                "doc_id": {
                    "type": "batch",
                    "fact_boundaries": [...],
                    "text": "..."
                }  # 批量模式
            }
        """
        doc_to_fact = {}
        
        if "id" in documents.columns and "metadata" in documents.columns:
            for _, row in documents.iterrows():
                doc_id = str(row["id"])
                meta = row.get("metadata")
                
                try:
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    
                    if isinstance(meta, dict):
                        # 检查是否是批量模式
                        if "fact_boundaries" in meta:
                            # 批量模式：记录fact边界信息
                            doc_to_fact[doc_id] = {
                                "type": "batch",
                                "fact_boundaries": meta["fact_boundaries"],
                                "conversation_id": meta.get("conversation_id"),
                                "text": str(row.get("text", ""))
                            }
                        elif "fact_id" in meta:
                            # 单fact模式
                            fact_id = meta.get("fact_id")
                            if fact_id is not None:
                                doc_to_fact[doc_id] = int(fact_id)
                except Exception as e:
                    pass
        
        return doc_to_fact
    
    def _build_tu_to_doc_map(self, text_units: pd.DataFrame) -> Dict[str, str]:
        """构建 text_unit_id -> document_id 映射"""
        tu_to_doc = {}
        
        doc_id_col = None
        for c in ["document_ids", "document_id", "documents"]:
            if c in text_units.columns:
                doc_id_col = c
                break
        
        if doc_id_col:
            for _, row in text_units.iterrows():
                tu_id = str(row["id"])
                doc_ids = self._parse_list(row.get(doc_id_col))
                if doc_ids:
                    tu_to_doc[tu_id] = str(doc_ids[0])
        
        return tu_to_doc
    
    def _build_community_facts_map(
        self,
        communities: pd.DataFrame,
        tu_to_doc: Dict[str, str],
        doc_to_fact: Dict[str, Any]
    ) -> Dict[int, Set[int]]:
        """构建 community -> fact_ids 映射"""
        community_facts = defaultdict(set)
        
        for _, row in communities.iterrows():
            comm_id = row.get("community")
            if comm_id is None:
                continue
            
            # 获取该社区的 text_unit_ids
            tu_ids_col = None
            for c in ["text_unit_ids", "text_units", "tu_ids"]:
                if c in communities.columns:
                    tu_ids_col = c
                    break
            
            if not tu_ids_col:
                continue
            
            tu_ids = self._parse_list(row.get(tu_ids_col))
            
            # 通过 text_unit_ids -> document_ids -> fact_ids
            for tu_id in tu_ids:
                doc_id = tu_to_doc.get(str(tu_id))
                if not doc_id:
                    continue
                
                doc_info = doc_to_fact.get(doc_id)
                if doc_info is None:
                    continue
                
                # 检查是否是批量模式
                if isinstance(doc_info, dict) and doc_info.get("type") == "batch":
                    # 批量模式：从XML标记中提取fact_ids
                    fact_ids = self._extract_fact_ids_from_xml(doc_info)
                    community_facts[int(comm_id)].update(fact_ids)
                elif isinstance(doc_info, int):
                    # 单fact模式
                    community_facts[int(comm_id)].add(doc_info)
        
        return community_facts
    
    def _extract_fact_ids_from_xml(self, doc_info: Dict) -> Set[int]:
        """
        从批量文档的XML标记中提取fact_ids
        
        解析 <fact id="123">...</fact> 标记
        """
        import re
        
        fact_ids = set()
        text = doc_info.get("text", "")
        
        # 匹配 <fact id="数字">
        pattern = r'<fact\s+id="(\d+)">'
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                fact_ids.add(int(match))
            except ValueError:
                pass
        
        return fact_ids
    
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
                # 可能是字符串形式的列表，如 "[id1, id2]"
                try:
                    import ast
                    return ast.literal_eval(val)
                except Exception:
                    return []
        return []
    
    def _save_markdown(self, mapping: Dict[str, Any], md_path: Path) -> None:
        """保存为 Markdown 格式"""
        total_communities = len(mapping)
        total_facts = sum(c["fact_count"] for c in mapping.values())
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Knowledge Graph Communities\n\n")
            f.write(f"总计 {total_communities} 个社区，{total_facts} 条 facts\n\n")
            f.write("---\n\n")
            
            def md_sort_key(k):
                parts = k.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    return int(parts[1])
                return 999
            
            for comm_key in sorted(mapping.keys(), key=md_sort_key):
                comm_data = mapping[comm_key]
                comm_id = comm_data['community_id']
                fact_count = comm_data['fact_count']
                report = comm_data.get('report', {})
                
                f.write(f"## Community {comm_id}\n\n")
                
                # 写入 report 信息
                if report:
                    if report.get('title'):
                        f.write(f"**标题**: {report['title']}\n\n")
                    if report.get('summary'):
                        f.write(f"**摘要**: {report['summary']}\n\n")
                    if report.get('rating') is not None:
                        f.write(f"**重要性评分**: {report['rating']:.2f}\n\n")
                
                f.write(f"**Facts数量**: {fact_count}\n\n")
                
                for fact in comm_data['facts']:
                    f.write(f"### Fact #{fact['fact_id']}\n\n")
                    f.write(f"{fact['content']}\n\n")
                
                f.write("---\n\n")
    
    def _print_statistics(self, mapping: Dict[str, Any]) -> None:
        """打印统计信息"""
        print("\n" + "="*60)
        print("社区统计摘要:")
        print("="*60)
        
        def summary_sort_key(k):
            parts = k.split('_')
            if len(parts) > 1:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
            return 999
        
        total_facts = 0
        uncategorized_facts = 0
        categorized_communities = 0
        
        for comm_key in sorted(mapping.keys(), key=summary_sort_key)[:10]:
            comm_data = mapping[comm_key]
            comm_id = comm_data['community_id']
            fact_count = comm_data['fact_count']
            total_facts += fact_count
            
            if comm_id == -1:
                uncategorized_facts = fact_count
                print(f"  Community {comm_id:>3}: {fact_count:>3} facts (未分类)")
            else:
                categorized_communities += 1
                print(f"  Community {comm_id:>3}: {fact_count:>3} facts")
        
        if len(mapping) > 10:
            print(f"  ... 还有 {len(mapping) - 10} 个社区")
        
        print(f"\n  ✓ 总计: {len(mapping)} 个社区, {total_facts} 条 facts")
        if uncategorized_facts > 0:
            coverage = (total_facts - uncategorized_facts) / total_facts * 100
            print(f"    - GraphRAG 覆盖: {total_facts - uncategorized_facts}/{total_facts} ({coverage:.1f}%)")
            print(f"    - 未分类: {uncategorized_facts} 条")

