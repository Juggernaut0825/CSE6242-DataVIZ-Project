"""
Bundle精装修模块
将存储导向的JSON转化为推理导向的JSON
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import re


def parse_datetime_to_date(timestamp: str) -> Optional[str]:
    """
    解析时间戳为YYYY-MM-DD格式
    
    支持多种格式:
    - "2023-05-08T14:30:00"
    - "1:56 pm on 8 May, 2023"
    - "2023-05-08 14:30:00"
    """
    if not timestamp:
        return None
    
    try:
        # 尝试ISO格式
        if 'T' in timestamp or '-' in timestamp[:10]:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        
        # LoCoMo格式: "1:56 pm on 8 May, 2023"
        match = re.search(r'(\d{1,2})\s+(\w+),?\s+(\d{4})', timestamp)
        if match:
            day, month_name, year = match.groups()
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12,
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9,
                'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = month_map.get(month_name, 1)
            return f"{year}-{month:02d}-{int(day):02d}"
        
        return None
    except:
        return None


def flatten_hop_facts(hop_facts: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """
    拍平hop_facts为简单的字符串列表
    
    输入格式:
    {
        "1hop": [
            {
                "fact_id": 123,
                "content": "...",
                "path": [...],
                "relation": "SUPPORT"
            }
        ]
    }
    
    输出格式:
    [
        "[SUPPORT] Caroline attended a support group. (1-hop)",
        "[CAUSE] Melanie felt inspired by Caroline's story. (2-hop)"
    ]
    """
    if not hop_facts:
        return []
    
    flattened = []
    
    for hop_key in sorted(hop_facts.keys()):
        hop_level = hop_key.replace('hop', '')
        facts_at_hop = hop_facts[hop_key]
        
        for fact in facts_at_hop:
            content = fact.get('content', '')
            relation = fact.get('relation', 'RELATED')
            
            # 格式: [关系类型] 内容 (N-hop)
            formatted = f"[{relation}] {content} ({hop_level}-hop)"
            flattened.append(formatted)
    
    return flattened


def decorate_fact_item(fact: Dict[str, Any]) -> Dict[str, Any]:
    """
    装修单个Fact项
    
    改动:
    1. 时间上浮: metadata.timestamp -> date (YYYY-MM-DD)
    2. 拍平hop_facts: 嵌套结构 -> 字符串数组
    3. 剔除噪音: 删除project_id, sample_id等（保留必需的score）
    """
    decorated = {
        'fact_id': fact['fact_id'],
        'content': fact['content'],
        'score': fact.get('score', 0.0)  # 保留score字段（Pydantic必需）
    }
    
    # 1. 提取并格式化时间
    metadata = fact.get('metadata', {})
    if metadata:
        timestamp = metadata.get('timestamp')
        date = parse_datetime_to_date(timestamp)
        if date:
            decorated['date'] = date
        
        # 清理metadata：只保留有意义的字段
        clean_metadata = {}
        useful_keys = ['username', 'session_id', 'participants']
        for key in useful_keys:
            if key in metadata and metadata[key]:
                clean_metadata[key] = metadata[key]
        
        if clean_metadata:
            decorated['metadata'] = clean_metadata
    
    # 2. 拍平hop_facts（如果有图谱扩展）
    hop_facts = fact.get('hop_facts')
    if hop_facts and isinstance(hop_facts, dict):
        context = flatten_hop_facts(hop_facts)
        if context:
            decorated['context'] = context
        # 删除原始的hop_facts（已拍平到context）
    else:
        decorated['hop_facts'] = None  # Pydantic允许None
    
    # 3. 保留图片URL（如果有）
    decorated['image_url'] = fact.get('image_url')  # 允许None
    
    return decorated


def decorate_conversation_item(conv: Dict[str, Any]) -> Dict[str, Any]:
    """
    装修单个Conversation项
    
    改动:
    1. 时间上浮
    2. 剔除噪音（保留必需的score）
    """
    decorated = {
        'conversation_id': conv['conversation_id'],
        'text': conv['text'],
        'score': conv.get('score', 0.0)  # 保留score字段（Pydantic必需）
    }
    
    # 提取时间并清理metadata
    metadata = conv.get('metadata', {})
    if metadata:
        timestamp = metadata.get('timestamp')
        date = parse_datetime_to_date(timestamp)
        if date:
            decorated['date'] = date
        
        # 清理metadata
        clean_metadata = {}
        useful_keys = ['username', 'session_id', 'participants']
        for key in useful_keys:
            if key in metadata and metadata[key]:
                clean_metadata[key] = metadata[key]
        
        if clean_metadata:
            decorated['metadata'] = clean_metadata
        else:
            decorated['metadata'] = None  # Pydantic允许None
    else:
        decorated['metadata'] = None
    
    return decorated


def decorate_topic_item(topic: Dict[str, Any]) -> Dict[str, Any]:
    """
    装修单个Topic项
    
    改动: 剔除噪音（保留必需的score）
    """
    return {
        'topic_id': topic['topic_id'],
        'title': topic['title'],
        'summary': topic['summary'],
        'score': topic.get('score', 0.0)  # 保留score字段（Pydantic必需）
    }


def decorate_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    装修整个Bundle
    
    对内部的Facts、Conversations、Topics进行精装修
    """
    decorated = {
        'bundle_id': bundle['bundle_id']
    }
    
    # 装修Facts
    if bundle.get('facts'):
        decorated['facts'] = [
            decorate_fact_item(fact) 
            for fact in bundle['facts']
        ]
    else:
        decorated['facts'] = []
    
    # 装修Conversations
    if bundle.get('conversations'):
        decorated['conversations'] = [
            decorate_conversation_item(conv) 
            for conv in bundle['conversations']
        ]
    else:
        decorated['conversations'] = []
    
    # 装修Topics
    if bundle.get('topics'):
        decorated['topics'] = [
            decorate_topic_item(topic) 
            for topic in bundle['topics']
        ]
    else:
        decorated['topics'] = []
    
    # 如果是精炼后的Bundle，保留related_memory（必需字段）
    if 'related_memory' in bundle:
        decorated['related_memory'] = bundle['related_memory']
    
    # quote是可选的
    if bundle.get('quote'):
        decorated['quote'] = bundle['quote']
    
    return decorated


def decorate_bundles(bundles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量装修Bundles
    
    这是对外的主要接口
    """
    return [decorate_bundle(bundle) for bundle in bundles]


def decorate_bundle_response(response: Dict[str, Any], enable: bool = True) -> Dict[str, Any]:
    """
    装修整个BundleQueryResponse
    
    Args:
        response: BundleQueryResponse的字典形式
        enable: 是否启用装修（默认True）
    
    Returns:
        装修后的response
    """
    if not enable:
        return response
    
    decorated_response = {
        'query': response['query'],
        'project_id': response['project_id'],
        'bundles': decorate_bundles(response.get('bundles', [])),
        'total_bundles': response.get('total_bundles', 0),
        'refined': response.get('refined', False)
    }
    
    # 保留short_term_memory（如果有）
    if response.get('short_term_memory'):
        decorated_response['short_term_memory'] = response['short_term_memory']
    
    # 保留recent_turns（如果有）
    if response.get('recent_turns'):
        decorated_response['recent_turns'] = response['recent_turns']
    
    # 保留graph_expansion信息（统计用）
    if response.get('graph_expansion'):
        decorated_response['graph_expansion'] = response['graph_expansion']
    
    return decorated_response


# 示例
if __name__ == '__main__':
    # 测试装修效果
    sample_bundle = {
        'bundle_id': 0,
        'facts': [
            {
                'fact_id': 123,
                'content': 'Caroline attended a support group.',
                'score': 0.95,
                'metadata': {
                    'timestamp': '1:56 pm on 8 May, 2023',
                    'project_id': 'test',
                    'sample_id': 'conv-26'
                },
                'hop_facts': {
                    '1hop': [
                        {
                            'fact_id': 456,
                            'content': 'The support group helped Caroline feel accepted.',
                            'relation': 'SUPPORT',
                            'path': [123, 456]
                        }
                    ],
                    '2hop': [
                        {
                            'fact_id': 789,
                            'content': 'Caroline gained confidence from the experience.',
                            'relation': 'CAUSE',
                            'path': [123, 456, 789]
                        }
                    ]
                }
            }
        ],
        'conversations': [
            {
                'conversation_id': 1,
                'text': 'Hey Mel! I went to a support group yesterday.',
                'score': 0.88,
                'metadata': {
                    'timestamp': '2023-05-08T13:56:00',
                    'session_id': 'session_1'
                }
            }
        ]
    }
    
    print("装修前:")
    import json
    print(json.dumps(sample_bundle, indent=2, ensure_ascii=False))
    
    print("\n装修后:")
    decorated = decorate_bundle(sample_bundle)
    print(json.dumps(decorated, indent=2, ensure_ascii=False))

