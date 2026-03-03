"""
显性关系提取模块
在 Fact 提取阶段直接识别事实之间的显性关系
"""
import requests
import json
from typing import List, Dict, Any, Optional


class ExplicitRelationExtractor:
    """显性关系提取器"""
    
    # 支持的显性关系类型
    RELATION_TYPES = {
        "Cause": "Causal relationship (A is the cause of B)",
        "Result": "Result relationship (A leads to B)",
        "Temporal": "Temporal relationship (A happens before/after B)",
        "Support": "Support relationship (A supports/proves B)",
        "Contradict": "Contradiction relationship (A contradicts B)",
        "Elaborate": "Elaboration relationship (A elaborates on B in detail)",
        "Condition": "Conditional relationship (If A then B)",
        "Purpose": "Purpose relationship (The purpose of A is B)",
        "Parallel": "Parallel relationship (A and B are parallel options, alternatives, or items at the same level)",
    }
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ):
        """
        初始化显性关系提取器
        
        Args:
            api_base: LLM API Base URL
            api_key: LLM API Key
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def extract_relations_from_facts(
        self,
        facts: List[str],
        conversation_text: str = None
    ) -> List[Dict[str, Any]]:
        """
        从 facts 列表中提取显性关系
        
        Args:
            facts: facts 列表（已提取的原子事实）
            conversation_text: 原始对话文本（可选，用于提供上下文）
        
        Returns:
            关系列表：[
                {
                    'source_fact_index': int,  # 源 fact 在列表中的索引
                    'target_fact_index': int,  # 目标 fact 在列表中的索引
                    'relation_type': str,      # 关系类型
                    'confidence': float,       # 置信度 (0.0-1.0)
                    'explanation': str         # 关系说明（可选）
                },
                ...
            ]
        """
        if len(facts) < 2:
            # 少于2条fact，无需提取关系
            return []
        
        # 构建 prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(facts, conversation_text)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用 LLM
        response_text = self._call_llm(messages)
        
        # 解析响应
        relations = self._parse_relations(response_text, len(facts))
        
        return relations
    
    def _build_system_prompt(self) -> str:
        """构建系统 prompt"""
        relation_types_text = "\n".join([
            f"- {key}: {value}"
            for key, value in self.RELATION_TYPES.items()
        ])
        
        return f"""You are an expert in identifying EXPLICIT logical relationships between facts.

**Your Task**: Analyze a list of atomic facts and identify direct, explicit relationships between them.

**Supported Relationship Types**:
{relation_types_text}

**Key Principles**:
1. **Only extract EXPLICIT relationships** - the relationship must be directly stated or strongly implied by the facts themselves
2. **Ignore vague or speculative connections** - do not infer relationships based on general knowledge
3. **Focus on high-confidence relationships** (confidence >= 0.7)
4. **Directional clarity** - clearly identify which fact is the source and which is the target
5. **Atomicity** - one relationship describes one logical connection

**Output Format** (JSON array):
[
  {{
    "source_fact_index": 0,
    "target_fact_index": 2,
    "relation_type": "Cause",
    "confidence": 0.9,
    "explanation": "Fact 0 explicitly states the cause of the event described in Fact 2"
  }}
]

If no explicit relationships exist, return an empty array: []"""
    
    def _build_user_prompt(
        self,
        facts: List[str],
        conversation_text: Optional[str]
    ) -> str:
        """构建用户 prompt"""
        # 构建 facts 列表
        facts_text = "\n".join([
            f"[{i}] {fact}"
            for i, fact in enumerate(facts)
        ])
        
        prompt = f"""**Facts List** (Total: {len(facts)}):
{facts_text}

Please identify explicit relationships between these facts."""
        
        # 如果提供了原始对话，添加上下文
        if conversation_text:
            context_preview = conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text
            prompt += f"\n\n**Original Context** (for reference):\n{context_preview}"
        
        return prompt
    
    def _call_llm(self, messages: List[dict]) -> str:
        """调用 LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"]
    
    def _parse_relations(
        self,
        response_text: str,
        num_facts: int
    ) -> List[Dict[str, Any]]:
        """
        解析 LLM 返回的关系列表
        
        Args:
            response_text: LLM 响应文本
            num_facts: facts 总数（用于验证索引）
        
        Returns:
            解析后的关系列表
        """
        try:
            # 尝试提取 JSON（可能被 markdown 包裹）
            content = response_text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 解析 JSON
            relations = json.loads(content)
            
            # 如果返回的是对象而不是数组，尝试提取
            if isinstance(relations, dict):
                if 'relations' in relations:
                    relations = relations['relations']
                elif 'result' in relations:
                    relations = relations['result']
                else:
                    relations = []
            
            if not isinstance(relations, list):
                return []
            
            # 验证和过滤关系
            valid_relations = []
            for rel in relations:
                # 验证必需字段
                if not all(key in rel for key in ['source_fact_index', 'target_fact_index', 'relation_type']):
                    continue
                
                # 验证索引范围
                if not (0 <= rel['source_fact_index'] < num_facts and 
                        0 <= rel['target_fact_index'] < num_facts):
                    continue
                
                # 验证关系类型
                if rel['relation_type'] not in self.RELATION_TYPES:
                    continue
                
                # 验证置信度（如果没有提供，默认0.8）
                confidence = rel.get('confidence', 0.8)
                if confidence < 0.7:
                    continue
                
                # 添加到有效关系列表
                valid_relations.append({
                    'source_fact_index': rel['source_fact_index'],
                    'target_fact_index': rel['target_fact_index'],
                    'relation_type': rel['relation_type'],
                    'confidence': confidence,
                    'explanation': rel.get('explanation', '')
                })
            
            return valid_relations
        
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON 解析失败: {e}")
            print(f"  响应内容: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"  ⚠️  解析关系失败: {e}")
            return []
    
    @staticmethod
    def format_relations_for_neo4j(
        relations: List[Dict[str, Any]],
        fact_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        将关系格式化为 Neo4j 存储格式
        
        Args:
            relations: 提取的关系列表（使用索引）
            fact_ids: 实际的 fact_id 列表
        
        Returns:
            Neo4j 格式的关系列表：[
                {
                    'source_fact_id': int,
                    'target_fact_id': int,
                    'relation_type': str,
                    'confidence': float,
                    'source': 'explicit'  # 标记为显性关系
                },
                ...
            ]
        """
        neo4j_relations = []
        
        for rel in relations:
            source_idx = rel['source_fact_index']
            target_idx = rel['target_fact_index']
            
            # 确保索引有效
            if source_idx >= len(fact_ids) or target_idx >= len(fact_ids):
                continue
            
            neo4j_relations.append({
                'source_fact_id': fact_ids[source_idx],
                'target_fact_id': fact_ids[target_idx],
                'relation_type': rel['relation_type'],
                'confidence': rel['confidence'],
                'explanation': rel.get('explanation', ''),
                'source': 'explicit'  # 标记来源
            })
        
        return neo4j_relations

