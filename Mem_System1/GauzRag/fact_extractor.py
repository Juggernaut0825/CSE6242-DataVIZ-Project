"""
Facts 提取模块（含显性关系提取）
从原始文档中提取结构化的 facts 并识别显性关系
"""
import requests
import json
import re
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


class FactExtractor:
    """Facts 提取器（一次性提取 Facts 和显性关系）"""
    
    # 支持的显性关系类型
    RELATION_TYPES = {
        "Cause", "Result", "Temporal", "Support", "Contradict", 
        "Elaborate", "Condition", "Purpose", "Parallel"
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
        初始化 Facts 提取器
        
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
    
    def extract_from_text(
        self, 
        text: str, 
        metadata: dict = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        从文本中提取 facts 和显性关系（一次性LLM调用）
        
        Args:
            text: 输入文本
            metadata: 元数据（可选），包含额外的上下文信息
        
        Returns:
            (facts_text, relations) 元组：
            - facts_text: facts 文本（每行一条）
            - relations: 显性关系列表
        """
        # 构建合并的 prompt（同时提取 facts 和关系）
        system_prompt = (
            "You are an expert in **Atomic Fact Extraction and Relationship Analysis**.\n\n"
            
            "**Phase 1: Extract Atomic Facts**\n"
            "⚠️ Core Principle: **Atomicity**\n"
            "- A fact must contain **only one** core piece of information.\n"
            "- **Strictly avoid compound sentences.** If a sentence contains 'and', 'but', or relative clauses, SPLIT it.\n"
            "- **Extract modifiers as separate facts.** Adjectives and identities often represent independent attributes.\n\n"
            
            "⚠️ Extraction Strategies:\n"
            "1. **Decompose Compound Actions:**\n"
            "   - Input: 'John moved to Tokyo and started a new job.'\n"
            "   - Output: John moved to Tokyo. | John started a new job.\n"
            "2. **Decompose Entity Properties:**\n"
            "   - Input: 'I drove my red Ferrari.'\n"
            "   - Output: The user drove a Ferrari. | The Ferrari is red.\n\n"
            
            "Requirements for Each Fact:\n"
            "- Be a complete statement with subject, predicate, and object.\n"
            "- **Replace pronouns** (he, she, it) with specific entity names.\n"
            "- Be as concise as possible.\n\n"
            
            "⚠️ Time Reference Resolution:\n"
            "- When you encounter relative time expressions (like 'yesterday', 'last year'), "
            "KEEP the original expression and ADD the absolute time in parentheses.\n"
            "- Use the reference time from [Context] to calculate the absolute time.\n"
            "- ⚠️ CRITICAL: Format dates in ENGLISH ONLY (e.g., 'May 7, 2023', '2023', 'June 2023').\n"
            "- Examples: 'yesterday' → 'yesterday (7 May 2023)' | 'last year' → 'last year (2022)'\n\n"
            
            "**Phase 2: Identify Explicit Relationships**\n"
            "After extracting facts, analyze them and identify direct, explicit relationships.\n\n"
            
            "Supported Relationship Types:\n"
            "- Cause: Causal relationship (A is the cause of B)\n"
            "- Result: Result relationship (A leads to B)\n"
            "- Temporal: Temporal relationship (A happens before/after B)\n"
            "- Support: Support relationship (A supports/proves B)\n"
            "- Contradict: Contradiction relationship (A contradicts B)\n"
            "- Elaborate: Elaboration relationship (A elaborates on B)\n"
            "- Condition: Conditional relationship (If A then B)\n"
            "- Purpose: Purpose relationship (The purpose of A is B)\n"
            "- Parallel: Parallel relationship (A and B are alternatives/same level)\n\n"
            
            "Key Principles:\n"
            "- Only extract EXPLICIT relationships (confidence >= 0.7)\n"
            "- Do NOT infer relationships based on speculation or general knowledge\n"
            "- Clearly identify source and target facts\n\n"
            
            "**Output Format (JSON)**:\n"
            "{\n"
            '  "facts": [\n'
            '    "fact content 1",\n'
            '    "fact content 2"\n'
            "  ],\n"
            '  "relations": [\n'
            "    {\n"
            '      "source_fact_index": 0,\n'
            '      "target_fact_index": 1,\n'
            '      "relation_type": "Support",\n'
            '      "confidence": 0.85,\n'
            '      "explanation": "optional explanation"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "If no explicit relationships exist, set \"relations\" to an empty array: []"
        )
        
        # Build user prompt with metadata
        user_prompt = ""
        
        # Add concise context information if metadata exists
        if metadata:
            context_parts = []
            if metadata.get("username"):
                context_parts.append(f"Speaker: {metadata['username']}")
            if metadata.get("timestamp"):
                context_parts.append(f"Reference Time: {metadata['timestamp']}")
            
            if context_parts:
                user_prompt += "[Context] " + ", ".join(context_parts) + "\n\n"
        
        # Add document content to extract
        user_prompt += f"[Document]\n{text}\n\n"
        user_prompt += "Please extract atomic facts and identify explicit relationships between them."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用 LLM API（一次性获取 facts 和 relations）
        response_text = self._call_llm(messages)
        
        # 解析 JSON 响应
        facts_text, relations = self._parse_response(response_text)
        
        return facts_text, relations
    
    def extract_from_file(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """
        从文件中提取 facts 和显性关系
        
        Args:
            file_path: 文件路径
        
        Returns:
            (facts_text, relations) 元组
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.extract_from_text(text)
    
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
    
    def _parse_response(
        self, 
        response_text: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        解析 LLM 响应（JSON格式：包含facts和relations）
        
        Args:
            response_text: LLM 返回的文本
        
        Returns:
            (facts_text, relations) 元组
        """
        try:
            # 提取 JSON（可能被 markdown 包裹）
            content = response_text.strip()
            
            # 尝试移除 markdown 代码块标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # 尝试找到第一个代码块
                parts = content.split("```")
                if len(parts) >= 3:
                    content = parts[1].strip()
                    # 如果有语言标识符（如json），移除它
                    if content.startswith("json\n"):
                        content = content[5:]
            
            # 解析 JSON
            data = json.loads(content)
            
            # 提取 facts
            facts_list = data.get("facts", [])
            if not isinstance(facts_list, list):
                print(f"  ⚠️  'facts' 字段不是列表，使用空列表")
                facts_list = []
            
            # 转换为文本格式（每行一条）
            facts_text = "\n".join(facts_list)
            
            # 提取 relations
            relations = data.get("relations", [])
            if not isinstance(relations, list):
                print(f"  ⚠️  'relations' 字段不是列表，使用空列表")
                relations = []
            
            # 验证和过滤关系
            valid_relations = []
            num_facts = len(facts_list)
            
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
            
            return facts_text, valid_relations
        
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON 解析失败: {e}")
            print(f"  响应内容: {response_text[:300]}...")
            # 降级：尝试按行解析 facts（忽略关系）
            facts_text = self._fallback_parse_facts(response_text)
            return facts_text, []
        
        except Exception as e:
            print(f"  ⚠️  解析响应失败: {e}")
            facts_text = self._fallback_parse_facts(response_text)
            return facts_text, []
    
    def _fallback_parse_facts(self, text: str) -> str:
        """
        降级方案：从非JSON响应中提取 facts（按行解析）
        
        Args:
            text: 响应文本
        
        Returns:
            facts 文本（每行一条）
        """
        lines = []
        for line in text.strip().splitlines():
            line = line.strip()
            if line and not line.startswith(("```", "{", "}", "[", "]")):
                # 移除可能的序号
                if line[0].isdigit() and ('.' in line[:5] or '、' in line[:5]):
                    line = line.split('.', 1)[-1].split('、', 1)[-1].strip()
                if line:
                    lines.append(line)
        return "\n".join(lines)
    
    @staticmethod
    def parse_facts(facts_text: str) -> List[str]:
        """
        解析 facts 文本为列表
        
        Args:
            facts_text: facts 文本（每行一条）
        
        Returns:
            facts 列表
        """
        facts = []
        for line in facts_text.strip().splitlines():
            line = line.strip()
            if line:
                # 移除可能的序号
                if line[0].isdigit() and ('.' in line[:5] or '、' in line[:5]):
                    line = line.split('.', 1)[-1].split('、', 1)[-1].strip()
                facts.append(line)
        return facts
    
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
            Neo4j 格式的关系列表
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

