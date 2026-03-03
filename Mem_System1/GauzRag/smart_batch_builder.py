"""
智能批量图谱构建器（改进版）
既享受批量处理的速度优势，又保留fact级别的实体映射
"""
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
import re
import json


def build_smart_batch_documents(facts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    改进的批量文档构建：
    1. 按 conversation_id 分组
    2. 每条fact带标记，用于后续映射
    3. 在metadata中记录fact边界
    
    Args:
        facts: facts列表
    
    Returns:
        DataFrame for GraphRAG
    """
    # 按 conversation_id 分组
    conv_groups = defaultdict(list)
    for fact in facts:
        conv_id = fact.get('conversation_id', f"orphan_{fact['fact_id']}")
        conv_groups[conv_id].append(fact)
    
    records = []
    
    for conv_id, conv_facts in conv_groups.items():
        conversation_text = conv_facts[0].get('conversation_text', '')
        conv_length = len(conversation_text)
        
        # 策略：用带标记的facts
        # 每条fact前加上特殊标记，便于后续映射
        fact_blocks = []
        fact_boundaries = []  # 记录每条fact的字符位置
        current_pos = 0
        
        for i, fact in enumerate(conv_facts):
            fact_id = fact['fact_id']
            content = fact['content']
            
            # 构建带标记的fact块
            # 使用XML风格标记，明确边界
            fact_block = f"<fact id=\"{fact_id}\">{content}</fact>"
            fact_blocks.append(fact_block)
            
            # 记录边界
            start_pos = current_pos
            end_pos = current_pos + len(fact_block)
            fact_boundaries.append({
                "fact_id": fact_id,
                "start": start_pos,
                "end": end_pos,
                "length": end_pos - start_pos
            })
            current_pos = end_pos + 1  # +1 for newline
        
        # 合并文本
        merged_text = "\n".join(fact_blocks)
        
        # 如果有对话原文，也加进来作为上下文
        if conversation_text and conv_length < 2000:
            final_text = f"""[对话原文]
{conversation_text}

[结构化事实（带fact_id标记）]
{merged_text}
"""
            strategy = "full_text_with_markers"
        else:
            final_text = merged_text
            strategy = "facts_with_markers"
        
        records.append({
            "id": f"conversation_{conv_id}",
            "title": f"conversation_{conv_id}",
            "text": final_text,
            "creation_date": conv_facts[0].get("created_at", datetime.now(timezone.utc).isoformat()),
            "metadata": {
                "type": "batch",  # 标记为批量文档
                "conversation_id": conv_id,
                "fact_boundaries": fact_boundaries,  # 关键：记录fact边界
                "fact_count": len(conv_facts),
                "strategy": strategy
            }
        })
        
        print(f"  → 对话 {conv_id}: {len(conv_facts)}条facts, 策略={strategy}")
    
    return pd.DataFrame.from_records(records)


def map_entities_to_facts(
    entities_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    documents_df: pd.DataFrame
) -> Dict[int, List[str]]:
    """
    将GraphRAG提取的实体映射回具体的fact
    
    Args:
        entities_df: GraphRAG的entities表
        text_units_df: GraphRAG的text_units表
        documents_df: 原始文档表（包含fact_boundaries）
    
    Returns:
        {fact_id: [entity1, entity2, ...]}
    """
    fact_to_entities = defaultdict(list)
    
    print(f"[DEBUG] 开始映射实体到facts...")
    print(f"[DEBUG] 实体总数: {len(entities_df)}")
    print(f"[DEBUG] Text Units总数: {len(text_units_df)}")
    print(f"[DEBUG] 文档总数: {len(documents_df)}")
    
    # 遍历所有实体
    for idx, entity_row in entities_df.iterrows():
        entity_name = entity_row['title']  # 实体名称
        text_unit_ids = entity_row.get('text_unit_ids', [])
        
        if idx < 3:  # 只打印前3个
            print(f"[DEBUG] 实体 '{entity_name}': {len(text_unit_ids) if hasattr(text_unit_ids, '__len__') else '?'} text_units")
        
        # 找到实体所在的text_units
        for text_unit_id in text_unit_ids:
            text_unit = text_units_df[text_units_df['id'] == text_unit_id]
            if text_unit.empty:
                continue
            
            # 获取text_unit的文本内容
            text_content = text_unit.iloc[0]['text']
            doc_ids = text_unit.iloc[0].get('document_ids', [])
            
            # 遍历相关文档
            for doc_id in doc_ids:
                doc = documents_df[documents_df['id'] == doc_id]
                if doc.empty:
                    continue
                
                metadata = doc.iloc[0].get('metadata', {})
                
                # 处理 metadata 可能是字符串的情况（JSON）
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                elif not isinstance(metadata, dict):
                    metadata = {}
                
                fact_boundaries = metadata.get('fact_boundaries', [])
                
                if idx == 0:  # 只为第一个实体打印详细信息
                    print(f"[DEBUG] 文档 {doc_id} metadata.type: {metadata.get('type', 'N/A')}")
                    print(f"[DEBUG] fact_boundaries类型: {type(fact_boundaries)}")
                    print(f"[DEBUG] fact_boundaries长度: {len(fact_boundaries) if hasattr(fact_boundaries, '__len__') else 'N/A'}")
                    if hasattr(fact_boundaries, '__len__') and len(fact_boundaries) > 0:
                        print(f"[DEBUG] fact_boundaries[0]: {fact_boundaries[0] if len(fact_boundaries) > 0 else 'empty'}")
                
                # 处理 NumPy 数组/pandas Series 的情况
                if fact_boundaries is None or (hasattr(fact_boundaries, '__len__') and len(fact_boundaries) == 0):
                    if idx == 0:
                        print(f"[DEBUG] fact_boundaries为空，跳过")
                    continue
                
                # 确保 fact_boundaries 是 Python 列表
                if hasattr(fact_boundaries, 'tolist'):
                    fact_boundaries = fact_boundaries.tolist()
                elif not isinstance(fact_boundaries, list):
                    fact_boundaries = list(fact_boundaries)
                
                # 解析XML标记，找到实体所在的fact
                # 简单方法：检查实体是否在某个fact的标记块中
                for boundary in fact_boundaries:
                    fact_id = boundary['fact_id']
                    # 在原文中找到这个fact的内容
                    pattern = f'<fact id="{fact_id}">(.*?)</fact>'
                    matches = re.findall(pattern, doc.iloc[0]['text'], re.DOTALL)
                    
                    if matches and entity_name in matches[0]:
                        fact_to_entities[fact_id].append(entity_name)
    
    # 去重
    for fact_id in fact_to_entities:
        fact_to_entities[fact_id] = list(set(fact_to_entities[fact_id]))
    
    print(f"[DEBUG] 映射完成，共 {len(fact_to_entities)} 个facts有实体")
    if len(fact_to_entities) > 0:
        sample_fact_id = list(fact_to_entities.keys())[0]
        print(f"[DEBUG] 示例: fact_{sample_fact_id} → {fact_to_entities[sample_fact_id]}")
    
    return dict(fact_to_entities)


# 使用示例
"""
# 1. 构建文档时使用带标记的方法
documents = build_smart_batch_documents(facts)

# 2. GraphRAG处理后，映射回fact
fact_to_entities = map_entities_to_facts(
    entities_df=outputs['entities'],
    text_units_df=outputs['text_units'],
    documents_df=documents
)

# 3. 结果
{
    7: ["王芳", "太阳帆计划", "数据分析"],  ✅ 精确对应
    9: ["王芳", "项目经理", "2023年10月"],
    ...
}
"""

