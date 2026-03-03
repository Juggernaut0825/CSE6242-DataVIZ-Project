"""
GauzRag 图谱构建完整流程
整合提取、存储、映射三大功能
支持 Leiden 社区检测优化
注：使用了 LightRAG 的实体提取 Prompt（后续将自定义）
"""
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase
import json
import asyncio
import time
import random
from .leiden_community_detector import LeidenCommunityDetector
from .semantic_topic_detector import SemanticTopicDetector


# ===== 实体提取 Prompt（源自 LightRAG，后续将自定义优化） =====

ENTITY_EXTRACTION_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: {entity_types}. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `<|#|>`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity<|#|>entity_name<|#|>entity_type<|#|>entity_description`

2.  **Delimiter Usage Protocol:**
    *   The `<|#|>` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity<|#|>Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity<|#|>Tokyo<|#|>location<|#|>Tokyo is the capital of Japan.`

3.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

4.  **Language:** The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

5.  **Completion Signal:** Output `<|COMPLETE|>` only after all entities have been extracted.

---Real Data---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

Output *only* the extracted list of entities. Do not include any explanations.
"""


# ===== 1. 实体提取器 =====

class GauzRagEntityExtractor:
    """
    使用实体提取 Prompt 提取实体（源自 LightRAG，后续将自定义）
    """
    
    def __init__(
        self,
        llm_api_key: str,
        llm_api_base: str,
        llm_model: str = "gpt-4o-mini",
        entity_types: Optional[List[str]] = None,
        language: str = "English"  # 统一使用英文
    ):
        self.client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_api_base)
        self.model = llm_model
        self.entity_types = entity_types or [
            "Person", "Organization", "Location", "Event",
            "Technology", "Model", "Algorithm", "Concept",
            "Method", "Technique", "System", "Tool"
        ]
        self.language = language
    
    async def aclose(self):
        """关闭异步客户端"""
        if hasattr(self, 'client') and self.client:
            await self.client.close()
    
    async def extract_from_fact(
        self,
        fact_content: str
    ) -> List[Dict[str, Any]]:
        """
        从 Fact 提取实体（已优化：不提取实体关系）
        
        Returns:
            entities: [{'name': str, 'type': str, 'description': str}, ...]
        """
        # 构建 Prompt
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            language=self.language,
            input_text=fact_content
        )
        
        # 调用 LLM
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500  # 减少 max_tokens（因为不输出关系了）
        )
        
        result = response.choices[0].message.content
        
        # 解析结果（只解析实体）
        entities = self._parse_extraction_result(result)
        
        return entities
    
    def _parse_extraction_result(
        self,
        result: str
    ) -> List[Dict[str, Any]]:
        """
        解析 LLM 返回结果（只解析实体）
        
        格式：
        entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
        """
        entities = []
        
        lines = result.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line or line == '<|COMPLETE|>':
                continue
            
            parts = line.split('<|#|>')
            
            if len(parts) < 2:
                continue
            
            if parts[0] == 'entity' and len(parts) >= 4:
                entities.append({
                    'name': parts[1].strip(),
                    'type': parts[2].strip(),
                    'description': parts[3].strip() if len(parts) > 3 else ''
                })
        
        return entities
    
    def normalize_entity_name(self, name: str) -> str:
        """
        标准化实体名称（可自定义实体合并规则）
        
        示例：
        - "Elon Musk" 和 "Elon" → "ELON_MUSK"
        - "马斯克" → "ELON_MUSK"
        """
        # 基础规则：转大写，替换空格为下划线
        normalized = name.strip().upper().replace(' ', '_')
        
        # 同义词映射（可扩展）
        synonyms = {
            'ELON': 'ELON_MUSK',
            'MUSK': 'ELON_MUSK',
            '马斯克': 'ELON_MUSK',
            # 添加更多同义词...
        }
        
        return synonyms.get(normalized, normalized)


# ===== 2. Neo4j 存储器 =====

class Neo4jEntityStore:
    """
    将提取的实体和关系存入 Neo4j
    """
    
    def __init__(self, uri: str, user: str, password: str, project_id: str = "default", database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = AsyncGraphDatabase.driver(
            uri, 
            auth=(user, password),
            max_connection_pool_size=50,  # 增加连接池大小（默认100）
            connection_acquisition_timeout=120.0  # 增加连接获取超时（默认60秒）
        )
        self.project_id = project_id
    
    async def close(self):
        """关闭连接"""
        await self.driver.close()
    
    async def create_constraints(self):
        """创建所有节点的约束和索引（优化：仅保留Fact和Topic）"""
        async with self.driver.session() as session:
            constraints_created = []
            constraints_failed = []
            
            # 1. Fact 节点唯一性约束（fact_id + project_id）
            try:
                await session.run(
                    """
                    CREATE CONSTRAINT fact_unique IF NOT EXISTS
                    FOR (f:Fact)
                    REQUIRE (f.fact_id, f.project_id) IS UNIQUE
                    """
                )
                constraints_created.append("fact_unique")
            except Exception as e:
                constraints_failed.append(("fact_unique", str(e)))
            
            # 2. Topic 节点唯一性约束（topic_id + project_id）
            try:
                await session.run(
                    """
                    CREATE CONSTRAINT topic_unique IF NOT EXISTS
                    FOR (t:Topic)
                    REQUIRE (t.topic_id, t.project_id) IS UNIQUE
                    """
                )
                constraints_created.append("topic_unique")
            except Exception as e:
                constraints_failed.append(("topic_unique", str(e)))
            
            # 3. 创建全文索引（用于搜索 Topic Summary）
            try:
                await session.run(
                    """
                    CREATE FULLTEXT INDEX topic_search IF NOT EXISTS
                    FOR (t:Topic)
                    ON EACH [t.title, t.summary]
                    """
                )
                constraints_created.append("topic_search")
            except Exception as e:
                constraints_failed.append(("topic_search", str(e)))
            
            # 4. 为Fact.entities创建索引（加速共享实体查询）
            try:
                await session.run(
                    """
                    CREATE INDEX fact_entities_index IF NOT EXISTS
                    FOR (f:Fact)
                    ON (f.entities)
                    """
                )
                constraints_created.append("fact_entities_index")
            except Exception as e:
                constraints_failed.append(("fact_entities_index", str(e)))
            
            # 输出结果
            if constraints_created:
                print(f"  ✓ 约束创建成功: {', '.join(constraints_created)}")
            if constraints_failed:
                print(f"  ⚠️ 约束创建失败:")
                for name, error in constraints_failed:
                    # 只显示错误的前100个字符
                    error_short = error[:100] + "..." if len(error) > 100 else error
                    print(f"    - {name}: {error_short}")
    
    async def create_topic_constraints(self):
        """[已废弃] 使用 create_constraints() 代替"""
        await self.create_constraints()
    
    async def store_fact_entities(
        self,
        fact_id: int,
        fact_content: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ):
        """
        存储 Fact 的实体和关系到 Neo4j
        
        构建图结构：
        - Fact 节点（带 project_id）
        - Entity 节点（带 project_id）
        - Fact-[:HAS_ENTITY]->Entity
        - Entity-[:RELATES_TO]->Entity
        """
        async with self.driver.session() as session:
            # 1. 创建 Fact 节点（带 project_id 隔离）
            await session.run(
                """
                MERGE (f:Fact {fact_id: $fact_id, project_id: $project_id})
                SET f.content = $content
                """,
                fact_id=fact_id,
                project_id=self.project_id,
                content=fact_content
            )
            
            # 2. 创建实体节点并连接到 Fact（带 project_id 隔离）
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:Entity {name: $name, project_id: $project_id})
                    SET e.type = $type, e.description = $description
                    WITH e
                    MATCH (f:Fact {fact_id: $fact_id, project_id: $project_id})
                    MERGE (f)-[:HAS_ENTITY]->(e)
                    """,
                    name=entity['name'],
                    project_id=self.project_id,
                    type=entity['type'],
                    description=entity['description'],
                    fact_id=fact_id
                )
            
            # 3. 创建实体之间的关系（在同一 project 内）
            for relation in relations:
                await session.run(
                    """
                    MATCH (s:Entity {name: $source, project_id: $project_id})
                    MATCH (t:Entity {name: $target, project_id: $project_id})
                    MERGE (s)-[r:RELATES_TO]->(t)
                    ON CREATE SET r.keywords = $keywords, r.description = $description, r.weight = 1
                    ON MATCH SET r.weight = r.weight + 1, r.description = COALESCE(r.description, '') + ' | ' + $description
                    """,
                    source=relation['source'],
                    target=relation['target'],
                    project_id=self.project_id,
                    keywords=relation['keywords'],
                    description=relation['description']
                )
    
    async def batch_store_facts_and_entities(
        self,
        extraction_results: List[Dict[str, Any]],
        unique_entities: Dict[str, Dict[str, str]]
    ):
        """
        批量存储Facts（Facts-Centric模式，实体作为属性）
        
        注意：此方法只创建 Fact 节点，语义关系由后续 LLM 分析创建
        
        Args:
            extraction_results: 每个Fact的提取结果
            unique_entities: 去重后的实体（用于构建 entities 属性）
        """
        async with self.driver.session() as session:
            # 批量创建Fact节点（entities作为属性）
            print(f"    - 创建 {len(extraction_results)} 个Fact节点（含实体属性）...")
            for result in extraction_results:
                # 提取实体名称（只存储名称列表）
                entity_names = [e['name'] for e in result['entities']]
                
                # 准备 Fact 属性（包括 conversation_id 用于后续删除）
                fact_properties = {
                    "fact_id": result['fact_id'],
                    "project_id": self.project_id,
                    "content": result['content'],
                    "entities": entity_names
                }
                
                # 如果有 conversation_id，也存储（用于替换删除）
                if 'conversation_id' in result:
                    fact_properties["conversation_id"] = result['conversation_id']
                
                await session.run(
                    """
                    MERGE (f:Fact {fact_id: $fact_id, project_id: $project_id})
                    SET f.content = $content,
                        f.entities = $entities,
                        f.conversation_id = $conversation_id
                    """,
                    fact_id=fact_properties['fact_id'],
                    project_id=fact_properties['project_id'],
                    content=fact_properties['content'],
                    entities=fact_properties['entities'],
                    conversation_id=fact_properties.get('conversation_id')
                )
            
            print(f"    ✓ Fact节点创建完成（语义关系将在Phase 5由LLM分析创建）")
    
    async def get_fact_entities(self, fact_id: int) -> List[str]:
        """获取 Fact 关联的实体（按 project_id 隔离）"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (f:Fact {fact_id: $fact_id, project_id: $project_id})-[:HAS_ENTITY]->(e:Entity)
                RETURN e.name AS name
                """,
                fact_id=fact_id,
                project_id=self.project_id
            )
            
            entities = []
            async for record in result:
                entities.append(record['name'])
            
            return entities
    
    async def get_all_fact_entities(self) -> Dict[int, List[str]]:
        """获取所有 Fact 的实体映射（按 project_id 隔离）"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e:Entity)
                RETURN f.fact_id AS fact_id, collect(e.name) AS entities
                """,
                project_id=self.project_id
            )
            
            fact_to_entities = {}
            async for record in result:
                fact_to_entities[record['fact_id']] = record['entities']
            
            return fact_to_entities
    
    async def get_entity_graph_stats(self) -> Dict[str, Any]:
        """获取实体图统计（按 project_id 隔离）"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {project_id: $project_id})
                WITH count(e) AS entity_count
                MATCH (e1:Entity {project_id: $project_id})-[r:RELATES_TO]->(e2:Entity {project_id: $project_id})
                RETURN entity_count, count(r) AS relation_count
                """,
                project_id=self.project_id
            )
            
            record = await result.single()
            
            return {
                'entity_count': record['entity_count'] if record else 0,
                'relation_count': record['relation_count'] if record else 0
            }
    
    # ===== Fact Relations 管理 =====
    
    async def build_fact_relations(
        self, 
        batch_size: int = 1000,
        use_community_optimization: bool = False,
        min_shared_entities: int = 1
    ) -> Dict[str, int]:
        """
        通过共享实体自动建立 Fact-Fact 关系
        
        策略：
        - 普通模式：全局匹配（O(n²)）
        - 社区优化模式：只在同一社区内匹配（O(k²)，k << n）
        
        Args:
            batch_size: 批处理大小
            use_community_optimization: 是否启用社区优化
            min_shared_entities: 最小共享实体数
        
        Returns:
            {"nodes": fact_count, "edges": relation_count, "mode": str}
        """
        mode = "community_optimized" if use_community_optimization else "global"
        
        async with self.driver.session() as session:
            if use_community_optimization:
                # 社区优化模式：只在同一社区内建立关系
                print("  🚀 使用社区优化模式构建关系")
                
                await session.run(
                    """
                    // 找到同一社区内的 Facts
                    MATCH (e1:Entity {project_id: $project_id})
                    WHERE e1.community_id IS NOT NULL
                    MATCH (f1:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e1)
                    
                    // 找到同社区内的其他 Facts（通过实体）
                    MATCH (e2:Entity {project_id: $project_id, community_id: e1.community_id})
                    MATCH (f2:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e2)
                    WHERE f1.fact_id < f2.fact_id
                    
                    // 计算共享实体
                    WITH f1, f2, collect(DISTINCT e2.name) AS shared_entities
                    WHERE size(shared_entities) >= $min_shared
                    
                    // 创建关系
                    MERGE (f1)-[r:RELATED_TO]-(f2)
                    SET r.shared_entities = shared_entities,
                        r.weight = size(shared_entities),
                        r.project_id = $project_id,
                        r.source = 'community_optimized'
                    """,
                    project_id=self.project_id,
                    min_shared=min_shared_entities
                )
            else:
                # 全局模式：匹配所有 Facts
                print("  ⚠️  使用全局模式构建关系（可能较慢）")
                
                await session.run(
                    """
                    MATCH (f1:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e:Entity)
                    MATCH (f2:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e)
                    WHERE f1.fact_id < f2.fact_id
                    WITH f1, f2, collect(DISTINCT e.name) AS shared_entities
                    WHERE size(shared_entities) >= $min_shared
                    MERGE (f1)-[r:RELATED_TO]-(f2)
                    SET r.shared_entities = shared_entities,
                        r.weight = size(shared_entities),
                        r.project_id = $project_id,
                        r.source = 'global'
                    """,
                    project_id=self.project_id,
                    min_shared=min_shared_entities
                )
            
            # 2. 获取统计
            result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                WITH count(f) AS fact_count
                MATCH (f1:Fact {project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                RETURN fact_count, count(DISTINCT r) AS relation_count
                """,
                project_id=self.project_id
            )
            
            record = await result.single()
            
            return {
                'nodes': record['fact_count'] if record else 0,
                'edges': record['relation_count'] if record else 0,
                'mode': mode
            }
    
    async def get_fact_relations(self, fact_id: int, max_relations: int = 50) -> List[Dict[str, Any]]:
        """
        获取指定 Fact 的所有相关 Facts
        
        Args:
            fact_id: Fact ID
            max_relations: 最大返回数量
        
        Returns:
            [{"fact_id": int, "content": str, "shared_entities": [str], "weight": int}]
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (f1:Fact {fact_id: $fact_id, project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                RETURN f2.fact_id AS fact_id, 
                       f2.content AS content,
                       r.shared_entities AS shared_entities,
                       r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT $max_relations
                """,
                fact_id=fact_id,
                project_id=self.project_id,
                max_relations=max_relations
            )
            
            relations = []
            async for record in result:
                relations.append({
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'shared_entities': record['shared_entities'],
                    'weight': record['weight']
                })
            
            return relations
    
    async def get_all_fact_relations(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        获取所有 Fact 的关系（用于构建完整关系图）
        
        Returns:
            {
                fact_id: [{"fact_id": int, "shared_entities": [str], "weight": int}]
            }
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (f1:Fact {project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                WHERE f1.fact_id < f2.fact_id  // 避免重复
                RETURN f1.fact_id AS source_id,
                       f2.fact_id AS target_id,
                       r.shared_entities AS shared_entities,
                       r.weight AS weight
                """,
                project_id=self.project_id
            )
            
            fact_relations = defaultdict(list)
            async for record in result:
                source_id = record['source_id']
                target_id = record['target_id']
                relation_data = {
                    'fact_id': target_id,
                    'shared_entities': record['shared_entities'],
                    'weight': record['weight']
                }
                
                # 无向图，双向存储
                fact_relations[source_id].append(relation_data)
                fact_relations[target_id].append({
                    'fact_id': source_id,
                    'shared_entities': record['shared_entities'],
                    'weight': record['weight']
                })
            
            return dict(fact_relations)
    
    async def get_candidate_facts(
        self, 
        fact_id: int,
        max_candidates: int = 20,
        use_community_optimization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        通过共享实体快速查找候选 Facts
        
        Args:
            fact_id: Fact ID
            max_candidates: 最大候选数
            use_community_optimization: 是否使用社区优化（默认 True）
        
        Returns:
            候选 Facts 列表
        """
        # 如果启用社区优化，首先尝试基于社区查找
        if use_community_optimization:
            try:
                # 1. 获取 Fact 的实体（从属性读取）
                async with self.driver.session() as session:
                    result = await session.run(
                        """
                        MATCH (f:Fact {fact_id: $fact_id, project_id: $project_id})
                        RETURN f.entities AS entities
                        """,
                        fact_id=fact_id,
                        project_id=self.project_id
                    )
                    record = await result.single()
                    entities = record['entities'] if record and record['entities'] else []
                
                if entities:
                    # 2. 使用 LeidenCommunityDetector 进行社区优化查找
                    detector = LeidenCommunityDetector(
                        uri=self.uri,
                        user=self.user,
                        password=self.password,
                        database=self.database
                    )
                    
                    try:
                        candidate_ids = detector.get_candidate_facts_by_community(
                            new_fact_id=fact_id,
                            entities=entities,
                            project_id=self.project_id,
                            max_candidates=max_candidates
                        )
                        
                        # 3. 获取候选 Facts 的详细信息（从属性读取）
                        if candidate_ids:
                            async with self.driver.session() as session:
                                result = await session.run(
                                    """
                                    MATCH (f:Fact {project_id: $project_id})
                                    WHERE f.fact_id IN $candidate_ids
                                      AND f.entities IS NOT NULL
                                    RETURN f.fact_id AS fact_id,
                                           f.content AS content,
                                           f.entities AS entities,
                                           f.entities AS shared_entities
                                    """,
                                    project_id=self.project_id,
                                    candidate_ids=candidate_ids
                                )
                                
                                candidates = []
                                async for record in result:
                                    candidates.append({
                                        'fact_id': record['fact_id'],
                                        'content': record['content'],
                                        'shared_entities': record['shared_entities']
                                    })
                                
                                return candidates
                    finally:
                        detector.close()
            
            except Exception as e:
                print(f"  ⚠️  社区优化查找失败，回退到全局查找: {e}")
        
        # 回退到全局查找（基于 Fact.entities 属性）
        async with self.driver.session() as session:
            result = await session.run(
                """
                // 找到新 Fact 的实体
                MATCH (f1:Fact {fact_id: $fact_id, project_id: $project_id})
                WHERE f1.entities IS NOT NULL AND size(f1.entities) > 0
                
                // 通过实体找到其他 Facts（基于 entities 属性）
                MATCH (f2:Fact {project_id: $project_id})
                WHERE f2.entities IS NOT NULL 
                  AND f1.fact_id <> f2.fact_id
                  AND any(e IN f1.entities WHERE e IN f2.entities)
                
                // 计算共享实体
                WITH f2, 
                     [e IN f1.entities WHERE e IN f2.entities] AS shared_entities
                
                // 按共享数量排序，取前 N 个
                RETURN f2.fact_id AS fact_id,
                       f2.content AS content,
                       f2.entities AS entities,
                       shared_entities,
                       size(shared_entities) AS weight
                ORDER BY weight DESC
                LIMIT $max_candidates
                """,
                fact_id=fact_id,
                project_id=self.project_id,
                max_candidates=max_candidates
            )
            
            candidates = []
            async for record in result:
                candidates.append({
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'entities': record['entities'],
                    'shared_entities': record['shared_entities']
                })
            
            return candidates
    
    async def add_semantic_relation(self, relation: Dict[str, Any]) -> None:
        """
        添加语义关系到 Neo4j
        
        Args:
            relation: {
                'new_fact_id': int,
                'target_fact_id': int,
                'relation_type': str,  # 因果/时序/支持等
                'confidence': float,
                'direction': str,
                'explanation': str
            }
        """
        async with self.driver.session() as session:
            # 解析方向
            direction = relation.get('direction', '')
            if '→' in direction or '->' in direction:
                parts = direction.replace('→', '->').split('->')
                source_ref = parts[0].strip()
                
                if 'new_fact' in source_ref:
                    source_id = relation['new_fact_id']
                    target_id = relation['target_fact_id']
                else:
                    source_id = relation['target_fact_id']
                    target_id = relation['new_fact_id']
            else:
                # 默认从 new_fact 指向 target_fact
                source_id = relation['new_fact_id']
                target_id = relation['target_fact_id']
            
            # 转换关系类型为英文（便于Neo4j）
            rel_type_map = {
                '支持': 'SUPPORT',
                '矛盾': 'CONTRADICT',
                '因果': 'CAUSE',
                '时序': 'TEMPORAL',
                '扩展': 'ELABORATE',
                '类比': 'ANALOGY',
                '条件': 'CONDITIONAL',
                '并列': 'PARALLEL',
                # 英文映射（兼容）
                'Support': 'SUPPORT',
                'Contradict': 'CONTRADICT',
                'Cause': 'CAUSE',
                'Temporal': 'TEMPORAL',
                'Elaborate': 'ELABORATE',
                'Analogy': 'ANALOGY',
                'Conditional': 'CONDITIONAL',
                'Parallel': 'PARALLEL'
            }
            
            rel_type_cn = relation.get('relation_type', '支持')
            rel_type_en = rel_type_map.get(rel_type_cn, 'SUPPORT')
            
            # 创建关系
            await session.run(
                f"""
                MATCH (source:Fact {{fact_id: $source_id, project_id: $project_id}})
                MATCH (target:Fact {{fact_id: $target_id, project_id: $project_id}})
                MERGE (source)-[r:{rel_type_en}]->(target)
                SET r.relation_type = $rel_type_cn,
                    r.confidence = $confidence,
                    r.project_id = $project_id
                """,
                source_id=source_id,
                target_id=target_id,
                rel_type_cn=rel_type_cn,
                confidence=relation.get('confidence', 0.8),
                project_id=self.project_id
            )
    
    async def get_fact_graph_stats(self) -> Dict[str, Any]:
        """获取 Fact 图统计"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                WITH count(f) AS nodes
                MATCH (f1:Fact {project_id: $project_id})-[r]-(f2:Fact {project_id: $project_id})
                WHERE f1.fact_id < f2.fact_id
                RETURN nodes, count(r) AS edges
                """,
                project_id=self.project_id
            )
            
            record = await result.single()
            return {
                'nodes': record['nodes'] if record else 0,
                'edges': record['edges'] if record else 0
            }
    
    # ========== Leiden 社区检测功能 ==========
    
    def run_leiden_community_detection(
        self,
        resolution: float = 1.0,
        min_community_size: int = 3
    ) -> Dict[str, Any]:
        """
        执行 Leiden 社区检测（同步方法）
        
        核心优势：
        - 降低 Fact 关系匹配复杂度：O(n²) → O(k²)
        - 优化图可视化：避免密集实体图
        - 加速检索：基于社区过滤
        
        Args:
            resolution: Leiden 分辨率参数（越大社区越多越小）
            min_community_size: 最小社区大小（过滤小社区）
        
        Returns:
            社区检测结果和统计信息
        """
        print(f"\n{'='*60}")
        print(f"🔍 执行 Leiden 社区检测 (project: {self.project_id})")
        print(f"{'='*60}")
        
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            result = detector.detect_communities(
                project_id=self.project_id,
                resolution=resolution,
                min_community_size=min_community_size
            )
            
            print(f"\n✓ 社区检测完成！")
            print(f"  - 总实体数：{result['total_entities']}")
            print(f"  - 社区数：{result['total_communities']}")
            print(f"  - 平均社区大小：{result['total_entities'] / max(result['total_communities'], 1):.1f}")
            
            return result
            
        finally:
            detector.close()
    
    def get_community_statistics(self) -> Dict[str, Any]:
        """
        获取社区统计信息（同步方法）
        
        Returns:
            统计信息字典
        """
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            return detector.get_community_statistics(self.project_id)
        finally:
            detector.close()
    
    def rebuild_fact_relations_by_community(
        self,
        min_shared_entities: int = 2
    ) -> Dict[str, int]:
        """
        基于社区重新构建 Fact 关系（同步方法）
        
        只在同一社区内建立关系，大幅降低时间复杂度
        
        Args:
            min_shared_entities: 最小共享实体数
        
        Returns:
            {'relations_created': int, 'communities_processed': int}
        """
        print(f"\n🔨 基于社区重新构建 Fact 关系")
        
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            result = detector.rebuild_fact_relations_by_community(
                project_id=self.project_id,
                min_shared_entities=min_shared_entities
            )
            
            print(f"✓ 关系重建完成！")
            print(f"  - 处理社区数：{result['communities_processed']}")
            print(f"  - 创建关系数：{result['relations_created']}")
            
            return result
            
        finally:
            detector.close()
    
    async def detect_fact_topics(self, min_size: int = 3) -> List[Dict[str, Any]]:
        """
        使用向量聚类检测 Topics（方案3：语义聚类）
        
        优势：
        - 基于向量相似度，避免连通分量的"语义漂移"问题
        - LLM 总结质量最高（同 Topic 内语义高度一致）
        - 增量友好（新 Fact 只需一次向量搜索）
        
        Args:
            min_size: 最小 Topic 大小（默认 3）
        
        Returns:
            [
                {
                    'topic_id': int,
                    'facts': [{fact_id: int, content: str}, ...],
                    'size': int,
                    'title': str,
                    'summary': str
                },
                ...
            ]
        """
        print(f"\n{'='*60}")
        print(f"启用方案3：向量聚类 Topic 检测")
        print(f"{'='*60}")
        
        # 注意：这是一个异步方法，但 SemanticTopicDetector 是同步的
        # 需要在异步上下文中调用同步代码
        
        # 步骤1: 获取所有 Facts（从 Neo4j）
        async with self.driver.session() as session:
            facts_result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                RETURN f.fact_id AS fact_id, f.content AS content
                ORDER BY f.fact_id
                """,
                project_id=self.project_id
            )
            
            all_facts = []
            async for record in facts_result:
                all_facts.append({
                    'fact_id': record['fact_id'],
                    'content': record['content']
                })
        
        print(f"  ✓ 从 Neo4j 加载了 {len(all_facts)} 个 Facts")
        
        # 步骤2: 调用 SemanticTopicDetector
        # TODO: 实现语义聚类逻辑
        
        print(f"  ⚠️  方案3的完整实现请使用 pipeline.detect_topics_with_semantic_clustering()")
        return []
    
    # ===== Hybrid Topic 相关方法 =====
    
    async def link_facts_to_topic(
        self,
        fact_ids: List[int],
        topic_id: int,
        project_id: str
    ):
        """建立 Fact → Topic 连接"""
        async with self.driver.session() as session:
            await session.run(
                """
                UNWIND $fact_ids AS fact_id
                MATCH (f:Fact {fact_id: fact_id, project_id: $project_id})
                MERGE (t:Topic {topic_id: $topic_id, project_id: $project_id})
                MERGE (f)-[:BELONGS_TO]->(t)
                """,
                fact_ids=fact_ids,
                topic_id=topic_id,
                project_id=project_id
            )
    
    async def mark_facts_as_buffer(
        self,
        fact_ids: List[int],
        project_id: str
    ):
        """标记 Facts 状态为 BUFFER"""
        async with self.driver.session() as session:
            await session.run(
                """
                UNWIND $fact_ids AS fact_id
                MATCH (f:Fact {fact_id: fact_id, project_id: $project_id})
                SET f.status = 'BUFFER'
                """,
                fact_ids=fact_ids,
                project_id=project_id
            )
    
    async def find_connected_buffer_facts(
        self,
        fact_ids: List[int],
        project_id: str
    ) -> List[int]:
        """
        查找与给定 Facts 共享实体的 Buffer Facts
        
        这是核心的连通性检查！
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (new:Fact)-[:HAS_ENTITY]->(:Entity)<-[:HAS_ENTITY]-(buffered:Fact)
                WHERE new.fact_id IN $fact_ids
                  AND new.project_id = $project_id
                  AND buffered.project_id = $project_id
                  AND buffered.status = 'BUFFER'
                  AND NOT buffered.fact_id IN $fact_ids
                RETURN DISTINCT buffered.fact_id AS fact_id
                """,
                fact_ids=fact_ids,
                project_id=project_id
            )
            
            neighbor_ids = []
            async for record in result:
                neighbor_ids.append(record['fact_id'])
            
            return neighbor_ids
    
    async def get_facts_by_ids(
        self,
        fact_ids: List[int],
        project_id: str
    ) -> List[Dict[str, Any]]:
        """根据 IDs 获取 Facts 详细信息"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                UNWIND $fact_ids AS fact_id
                MATCH (f:Fact {fact_id: fact_id, project_id: $project_id})
                OPTIONAL MATCH (f)-[:HAS_ENTITY]->(e:Entity)
                RETURN f.fact_id AS fact_id,
                       f.content AS content,
                       collect(DISTINCT e.name) AS entities
                """,
                fact_ids=fact_ids,
                project_id=project_id
            )
            
            facts = []
            async for record in result:
                facts.append({
                    "fact_id": record['fact_id'],
                    "content": record['content'],
                    "entities": record['entities']
                })
            
            return facts
    
    async def create_topic_node(
        self,
        topic_id: int,
        project_id: str,
        title: str,
        summary: str
    ):
        """创建 Topic 节点"""
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (t:Topic {topic_id: $topic_id, project_id: $project_id})
                SET t.title = $title,
                    t.summary = $summary,
                    t.created_at = datetime()
                """,
                topic_id=topic_id,
                project_id=project_id,
                title=title,
                summary=summary
            )
    
    async def clear_buffer_status(
        self,
        fact_ids: List[int],
        project_id: str
    ):
        """清除 Buffer 状态标记"""
        async with self.driver.session() as session:
            await session.run(
                """
                UNWIND $fact_ids AS fact_id
                MATCH (f:Fact {fact_id: fact_id, project_id: $project_id})
                REMOVE f.status
                """,
                fact_ids=fact_ids,
                project_id=project_id
            )
    
    async def generate_topic_summaries(
        self, 
        topics: List[Dict[str, Any]],
        llm_client,
        model: str
    ) -> List[Dict[str, Any]]:
        """
        为连通分量生成 Topic Summary
        
        Args:
            topics: detect_fact_topics() 的返回值
            llm_client: OpenAI 客户端
            model: LLM 模型名称
        
        Returns:
            [
                {
                    'topic_id': 35392,
                    'fact_count': 7,
                    'fact_ids': [35392, 35393, ...],
                    'title': '北方冬小麦干旱应对',
                    'summary': '...',
                    'key_points': ['...', '...'],
                    'importance': 8.5
                },
                ...
            ]
        """
        summaries = []
        
        print(f"\n生成 {len(topics)} 个 Topic Summaries...")
        
        for i, topic in enumerate(topics, 1):
            print(f"  [{i}/{len(topics)}] Topic {topic['topic_id']} ({topic['size']} facts)...")
            
            # 构建 Prompt
            facts_text = "\n".join([
                f"{j+1}. {fact['content']}"
                for j, fact in enumerate(topic['facts'])
            ])
            
            prompt = f"""You are a Knowledge Summarization Expert. Generate a Topic Summary for the following semantically connected Facts.

【Facts Count】{topic['size']}

【Facts List (connected through causal, temporal, support and other semantic relations)】
{facts_text}

Generate a concise Topic Summary (JSON format):
{{
  "title": "Topic title (within 10 words, summarize the theme)",
  "summary": "Topic summary (100-200 words, describe the overall meaning and logical connections)",
  "key_points": ["Key point 1 (one sentence)", "Key point 2", "Key point 3"],
  "importance": 8.5
}}

Importance scoring criteria (0-10):
- Information richness, logical completeness, practical value
"""
            
            try:
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a Knowledge Summarization Expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"} if "gpt" in model.lower() else None
                )
                
                report_text = response.choices[0].message.content
                report = json.loads(report_text)
                
                summaries.append({
                    'topic_id': topic['topic_id'],
                    'fact_count': topic['size'],
                    'fact_ids': [f['fact_id'] for f in topic['facts']],
                    'title': report.get('title', f"Topic {topic['topic_id']}"),
                    'summary': report.get('summary', ''),
                    'key_points': report.get('key_points', []),
                    'importance': report.get('importance', 5.0)
                })
                
            except Exception as e:
                print(f"    ⚠️  生成 Summary 失败: {str(e)}")
                summaries.append({
                    'topic_id': topic['topic_id'],
                    'fact_count': topic['size'],
                    'fact_ids': [f['fact_id'] for f in topic['facts']],
                    'title': f"Topic {topic['topic_id']}",
                    'summary': f"Summary 生成失败: {str(e)}",
                    'key_points': [],
                    'importance': 0
                })
        
        print(f"  ✓ 完成！")
        return summaries
    
    async def save_topics_to_neo4j(self, summaries: List[Dict[str, Any]]) -> None:
        """
        将 Topic Summaries 存储回 Neo4j
        
        创建结构：
        (:Topic {topic_id, title, summary, importance})
        (:Topic)-[:CONTAINS]->(:Fact)
        
        这样可以：
        1. 搜索 Summary 内容找到 Topic
        2. 从 Topic 导航到所有相关 Facts
        3. 可视化 Topic 层级结构
        
        Args:
            summaries: generate_topic_summaries() 的返回值
        """
        async with self.driver.session() as session:
            for summary in summaries:
                # 1. 创建/更新 Topic 节点
                await session.run(
                    """
                    MERGE (t:Topic {topic_id: $topic_id, project_id: $project_id})
                    SET t.title = $title,
                        t.summary = $summary,
                        t.importance = $importance,
                        t.fact_count = $fact_count,
                        t.key_points = $key_points,
                        t.updated_at = datetime()
                    """,
                    topic_id=summary['topic_id'],
                    project_id=self.project_id,
                    title=summary['title'],
                    summary=summary['summary'],
                    importance=summary['importance'],
                    fact_count=summary['fact_count'],
                    key_points=summary['key_points']
                )
                
                # 2. 建立 Topic -> Facts 关系
                await session.run(
                    """
                    MATCH (t:Topic {topic_id: $topic_id, project_id: $project_id})
                    
                    // 先删除旧的 CONTAINS 关系（因为连通分量可能变化）
                    OPTIONAL MATCH (t)-[old:CONTAINS]->()
                    DELETE old
                    
                    // 创建新的 CONTAINS 关系
                    WITH t
                    UNWIND $fact_ids AS fact_id
                    MATCH (f:Fact {fact_id: fact_id, project_id: $project_id})
                    MERGE (t)-[:CONTAINS]->(f)
                    """,
                    topic_id=summary['topic_id'],
                    project_id=self.project_id,
                    fact_ids=summary['fact_ids']
                )
        
        print(f"  ✓ {len(summaries)} 个 Topics 已存储到 Neo4j")
    
    async def get_all_topics(self) -> List[Dict[str, Any]]:
        """
        从 Neo4j 获取所有 Topics
        
        Returns:
            [
                {
                    'topic_id': int,
                    'title': str,
                    'summary': str,
                    'importance': float,
                    'fact_count': int,
                    'key_points': [str],
                    'updated_at': datetime
                },
                ...
            ]
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Topic {project_id: $project_id})
                RETURN t.topic_id AS topic_id,
                       t.title AS title,
                       t.summary AS summary,
                       t.importance AS importance,
                       t.fact_count AS fact_count,
                       t.key_points AS key_points,
                       t.updated_at AS updated_at
                ORDER BY t.importance DESC
                """,
                project_id=self.project_id
            )
            
            topics = []
            async for record in result:
                topics.append({
                    'topic_id': record['topic_id'],
                    'title': record['title'],
                    'summary': record['summary'],
                    'importance': record['importance'] or 5.0,
                    'fact_count': record['fact_count'] or 0,
                    'key_points': record['key_points'] or [],
                    'updated_at': record['updated_at']
                })
            
            return topics
    
    async def search_topics_fulltext(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用 Neo4j 全文索引搜索 Topics
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            Topics 列表（包含 score）
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                CALL db.index.fulltext.queryNodes('topic_search', $query)
                YIELD node AS topic, score
                WHERE topic.project_id = $project_id
                RETURN topic.topic_id AS topic_id,
                       topic.title AS title,
                       topic.summary AS summary,
                       topic.importance AS importance,
                       topic.fact_count AS fact_count,
                       topic.key_points AS key_points,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                query=query,
                project_id=self.project_id,
                top_k=top_k
            )
            
            topics = []
            async for record in result:
                topics.append({
                    'topic_id': record['topic_id'],
                    'title': record['title'],
                    'summary': record['summary'],
                    'importance': record['importance'] or 5.0,
                    'fact_count': record['fact_count'] or 0,
                    'key_points': record['key_points'] or [],
                    'score': record['score']
                })
            
            return topics
    
    async def get_facts_by_topic(self, topic_id: int) -> List[Dict[str, Any]]:
        """
        获取 Topic 包含的所有 Facts
        
        Args:
            topic_id: Topic ID
        
        Returns:
            Facts 列表
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Topic {topic_id: $topic_id, project_id: $project_id})-[:CONTAINS]->(f:Fact)
                RETURN f.fact_id AS fact_id,
                       f.content AS content
                ORDER BY f.fact_id
                """,
                topic_id=topic_id,
                project_id=self.project_id
            )
            
            facts = []
            async for record in result:
                facts.append({
                    'fact_id': record['fact_id'],
                    'content': record['content']
                })
            
            return facts
    
    async def expand_facts_by_semantic_relations(
        self,
        seed_fact_ids: List[int],
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        基于语义关系扩展 Facts（使用 CAUSE、TEMPORAL、SUPPORT 等关系）
        
        Args:
            seed_fact_ids: 种子 Fact IDs
            max_hops: 最大跳数（默认2跳）
            relation_types: 指定关系类型（如 ['CAUSE', 'TEMPORAL']），为 None 则使用全部 8 种关系
        
        Returns:
            扩展后的 Facts 列表：
            - fact_id: Fact ID
            - content: Fact 内容
            - entities: 实体列表
            - hop_distance: 跳数
            - reasoning_path: 完整路径（fact_id 列表）
            - relation_path: 关系类型路径
        """
        if not seed_fact_ids:
            return []
        
        # 构建关系类型字符串
        if relation_types:
            # 验证关系类型
            valid_relations = ['CAUSE', 'TEMPORAL', 'SUPPORT', 'ELABORATE', 'CONTRADICT', 'ANALOGY', 'CONDITIONAL']
            filtered_relations = [r.upper() for r in relation_types if r.upper() in valid_relations]
            if not filtered_relations:
                print(f"  ⚠️  无效的关系类型: {relation_types}，使用全部关系")
                relation_pattern = "CAUSE|TEMPORAL|SUPPORT|ELABORATE|CONTRADICT|ANALOGY|CONDITIONAL|PARALLEL"
            else:
                relation_pattern = "|".join(filtered_relations)
                print(f"  → 使用指定关系: {', '.join(filtered_relations)}")
        else:
            relation_pattern = "CAUSE|TEMPORAL|SUPPORT|ELABORATE|CONTRADICT|ANALOGY|CONDITIONAL|PARALLEL"
        
        async with self.driver.session() as session:
            # 使用无向边查询，兼容所有语义关系边（包括没有relation_type属性的边）
            cypher_query = f"""
                MATCH path = (seed:Fact)-[r*1..{max_hops}]-(expanded:Fact)
                WHERE seed.fact_id IN $seed_ids 
                  AND seed.project_id = $project_id
                  AND expanded.project_id = $project_id
                  AND NOT (expanded.fact_id IN $seed_ids)
                WITH DISTINCT expanded, 
                     length(path) AS hop_distance,
                     [n IN nodes(path) | n.fact_id] AS reasoning_path,
                     [rel IN relationships(path) | COALESCE(rel.relation_type, type(rel))] AS relation_types
                RETURN expanded.fact_id AS fact_id,
                       expanded.content AS content,
                       expanded.entities AS entities,
                       hop_distance,
                       reasoning_path,
                       relation_types
                ORDER BY hop_distance, expanded.fact_id
                """
            
            print(f"  → Cypher查询: 基于语义关系属性扩展（最大{max_hops}跳）")
            print(f"  → 种子IDs: {seed_fact_ids}")
            print(f"  → Project ID: {self.project_id}")
            
            # 先检查种子Facts是否有任何语义关系
            check_result = await session.run(
                """
                MATCH (seed:Fact)-[r]-(other:Fact)
                WHERE seed.fact_id IN $seed_ids 
                  AND seed.project_id = $project_id
                  AND other.project_id = $project_id
                  AND r.relation_type IS NOT NULL
                RETURN seed.fact_id, COALESCE(r.relation_type, type(r)) AS rel_type, other.fact_id
                LIMIT 10
                """,
                seed_ids=seed_fact_ids,
                project_id=self.project_id
            )
            
            relations_found = []
            async for record in check_result:
                relations_found.append({
                    'seed': record['seed.fact_id'],
                    'type': record['rel_type'],
                    'target': record['other.fact_id']
                })
            
            if relations_found:
                print(f"  → 种子Facts的关系样例（前10条）:")
                for rel in relations_found[:5]:
                    print(f"    · Fact {rel['seed']} -{rel['type']}-> Fact {rel['target']}")
            else:
                print(f"  ⚠️  种子Facts没有任何语义关系！")
            
            result = await session.run(
                cypher_query,
                seed_ids=seed_fact_ids,
                project_id=self.project_id
            )
            
            expanded_facts = []
            async for record in result:
                expanded_facts.append({
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'entities': record['entities'] or [],
                    'hop_distance': record['hop_distance'],
                    'reasoning_path': record['reasoning_path'],  # [seed_id, ..., expanded_id]
                    'relation_path': record['relation_types'],    # ['CAUSE', 'SUPPORT', ...]
                    'relations': record['relation_types']  # 兼容旧代码
                })
            
            return expanded_facts
    
    async def expand_facts_by_shared_entities(
        self,
        seed_fact_ids: List[int],
        max_hops: int = 1
    ) -> List[Dict[str, Any]]:
        """
        基于共享实体扩展 Facts（Facts-Centric模式）
        
        Args:
            seed_fact_ids: 种子 Fact IDs
            max_hops: 最大跳数（通过共享实体的传递关系）
        
        Returns:
            扩展后的 Facts 列表：
            - fact_id: Fact ID
            - content: Fact 内容
            - entities: 实体列表
            - hop_distance: 跳数
            - shared_entities: 与种子Fact共享的实体
        """
        if not seed_fact_ids:
            return []
        
        async with self.driver.session() as session:
            # 使用 SHARES_ENTITY 关系进行多跳扩展
            result = await session.run(
                f"""
                MATCH path = (seed:Fact)-[:SHARES_ENTITY*1..{max_hops}]-(expanded:Fact)
                WHERE seed.fact_id IN $seed_ids 
                  AND seed.project_id = $project_id
                  AND expanded.project_id = $project_id
                  AND NOT (expanded.fact_id IN $seed_ids)
                WITH DISTINCT expanded, 
                     length(path) AS hop_distance,
                     nodes(path) AS path_nodes,
                     relationships(path) AS rels
                RETURN expanded.fact_id AS fact_id,
                       expanded.content AS content,
                       expanded.entities AS entities,
                       hop_distance,
                       [n IN path_nodes | n.fact_id] AS reasoning_path,
                       [rel IN rels | rel.shared_entities] AS shared_entities_per_hop
                ORDER BY hop_distance, expanded.fact_id
                """,
                seed_ids=seed_fact_ids,
                project_id=self.project_id
            )
            
            expanded_facts = []
            async for record in result:
                # 提取共享实体（路径上的所有共享实体）
                shared_entities_per_hop = record['shared_entities_per_hop']
                all_shared_entities = []
                for entities_list in shared_entities_per_hop:
                    if entities_list:
                        all_shared_entities.extend(entities_list)
                
                expanded_facts.append({
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'entities': record['entities'] or [],
                    'hop_distance': record['hop_distance'],
                    'reasoning_path': record['reasoning_path'],
                    'shared_entities': list(set(all_shared_entities))  # 去重
                })
            
            return expanded_facts
    
    async def export_semantic_relations_to_json(self, output_path: Path) -> None:
        """导出语义关系到 JSON（用于社区检测等）"""
        async with self.driver.session() as session:
            # 获取所有 Facts
            facts_result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                RETURN f.fact_id AS fact_id,
                       f.content AS content
                """,
                project_id=self.project_id
            )
            
            nodes = {}
            async for record in facts_result:
                fact_id = str(record['fact_id'])
                nodes[fact_id] = {
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'entities': [],
                    'communities': []
                }
            
            # 获取所有语义关系
            relations_result = await session.run(
                """
                MATCH (f1:Fact {project_id: $project_id})-[r]-(f2:Fact {project_id: $project_id})
                WHERE type(r) IN ['SUPPORT', 'CONTRADICT', 'CAUSE', 'TEMPORAL', 'ELABORATE', 'ANALOGY', 'CONDITIONAL', 'PARALLEL']
                  AND f1.fact_id < f2.fact_id
                RETURN f1.fact_id AS source,
                       f2.fact_id AS target,
                       type(r) AS rel_type_en,
                       r.relation_type AS rel_type_cn,
                       r.confidence AS confidence,
                       r.explanation AS explanation
                """,
                project_id=self.project_id
            )
            
            edges = []
            async for record in relations_result:
                edges.append({
                    'source': str(record['source']),
                    'target': str(record['target']),
                    'relation_type': record['rel_type_cn'] or record['rel_type_en'],
                    'confidence': record['confidence'] or 0.8,
                    'explanation': record['explanation'] or ''
                })
            
            # 保存
            import json
            graph_data = {
                'nodes': nodes,
                'edges': edges,
                'metadata': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges)
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    async def get_fact_relation_stats(self) -> Dict[str, Any]:
        """获取 Fact Relations 统计"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                WITH count(f) AS total_facts
                MATCH (f1:Fact {project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                WITH total_facts, count(DISTINCT r) AS total_relations, avg(r.weight) AS avg_weight
                RETURN total_facts, total_relations, avg_weight
                """,
                project_id=self.project_id
            )
            
            record = await result.single()
            
            if not record:
                return {
                    'total_facts': 0,
                    'total_relations': 0,
                    'avg_weight': 0.0,
                    'avg_relations_per_fact': 0.0
                }
            
            total_facts = record['total_facts']
            total_relations = record['total_relations']
            
            return {
                'total_facts': total_facts,
                'total_relations': total_relations,
                'avg_weight': float(record['avg_weight']) if record['avg_weight'] else 0.0,
                'avg_relations_per_fact': (total_relations * 2) / total_facts if total_facts > 0 else 0.0
            }
    
    async def export_to_json(self, output_path: Path) -> None:
        """
        导出 Fact Relations 到 JSON 格式（用于 Community Detection 等遗留功能）
        
        Args:
            output_path: 输出 JSON 路径
        """
        import json
        from datetime import datetime
        
        async with self.driver.session() as session:
            # 获取所有 Facts
            facts_result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                OPTIONAL MATCH (f)-[:HAS_ENTITY]->(e:Entity)
                RETURN f.fact_id AS fact_id, 
                       f.content AS content,
                       collect(e.name) AS entities
                """,
                project_id=self.project_id
            )
            
            nodes = {}
            async for record in facts_result:
                fact_id = str(record['fact_id'])
                nodes[fact_id] = {
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'entities': record['entities']
                }
            
            # 获取所有关系
            relations_result = await session.run(
                """
                MATCH (f1:Fact {project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                WHERE f1.fact_id < f2.fact_id
                RETURN f1.fact_id AS source_id,
                       f2.fact_id AS target_id,
                       r.shared_entities AS shared_entities,
                       r.weight AS weight
                """,
                project_id=self.project_id
            )
            
            edges = []
            async for record in relations_result:
                edges.append({
                    'source': str(record['source_id']),
                    'target': str(record['target_id']),
                    'shared_entities': record['shared_entities'],
                    'weight': record['weight']
                })
            
            # 构建 JSON 结构
            graph_data = {
                'nodes': nodes,
                'edges': edges,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'source': 'neo4j',
                    'project_id': self.project_id
                }
            }
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)


# ===== 3. Entity→Facts 映射器 =====

class GauzRagEntityMapper:
    """
    维护 Entity → Facts 倒排索引
    用于快速构建 Facts 关系（避免 O(n²)）
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.entity_to_facts: Dict[str, Set[int]] = defaultdict(set)
        self.fact_to_entities: Dict[int, List[str]] = {}
    
    def add_fact(self, fact_id: int, entities: List[str]) -> None:
        """添加 Fact 到映射"""
        # 保存 fact → entities
        self.fact_to_entities[fact_id] = entities
        
        # 更新 entity → facts 倒排索引
        for entity in entities:
            entity_normalized = entity.strip().upper()
            self.entity_to_facts[entity_normalized].add(fact_id)
    
    def get_related_facts(self, entities: List[str]) -> Set[int]:
        """通过实体快速查找相关 Facts"""
        related_facts = set()
        
        for entity in entities:
            entity_normalized = entity.strip().upper()
            related_facts.update(self.entity_to_facts.get(entity_normalized, set()))
        
        return related_facts
    
    def get_fact_entities(self, fact_id: int) -> List[str]:
        """获取 Fact 的实体列表"""
        return self.fact_to_entities.get(fact_id, [])
    
    def get_shared_entities(self, fact_id1: int, fact_id2: int) -> List[str]:
        """获取两个 Facts 的共享实体"""
        entities1 = set(self.get_fact_entities(fact_id1))
        entities2 = set(self.get_fact_entities(fact_id2))
        return list(entities1 & entities2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取映射统计信息"""
        total_facts = len(self.fact_to_entities)
        total_entities = len(self.entity_to_facts)
        
        if total_entities > 0:
            avg_facts_per_entity = sum(len(facts) for facts in self.entity_to_facts.values()) / total_entities
        else:
            avg_facts_per_entity = 0
        
        return {
            'total_facts': total_facts,
            'total_entities': total_entities,
            'avg_facts_per_entity': avg_facts_per_entity
        }
    
    def save(self) -> None:
        """保存映射到文件"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 entity → facts
        entity_to_facts_file = self.output_dir / "entity_to_facts.json"
        with open(entity_to_facts_file, 'w', encoding='utf-8') as f:
            json.dump(
                {k: list(v) for k, v in self.entity_to_facts.items()},
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # 保存 fact → entities
        fact_to_entities_file = self.output_dir / "fact_to_entities.json"
        with open(fact_to_entities_file, 'w', encoding='utf-8') as f:
            json.dump(self.fact_to_entities, f, ensure_ascii=False, indent=2)
    
    def load(self) -> bool:
        """从文件加载映射"""
        entity_to_facts_file = self.output_dir / "entity_to_facts.json"
        fact_to_entities_file = self.output_dir / "fact_to_entities.json"
        
        if not entity_to_facts_file.exists() or not fact_to_entities_file.exists():
            return False
        
        with open(entity_to_facts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.entity_to_facts = {k: set(v) for k, v in data.items()}
        
        with open(fact_to_entities_file, 'r', encoding='utf-8') as f:
            self.fact_to_entities = json.load(f)
        
        return True


# ===== 4. GauzRag 图谱构建器（整合所有功能） =====

class GauzRagGraphBuilder:
    """
    GauzRag 图谱构建器
    注：实体提取prompt源自 LightRAG，后续将自定义优化
    
    完整流程：
    1. Fact 内容 → LightRAG Prompt 提取实体和关系
    2. 标准化实体名称
    3. 存入 Neo4j（Fact/Entity节点 + 关系）
    4. 维护 Entity→Facts 倒排索引
    5. 用于快速构建 Facts 关系图
    """
    
    def __init__(
        self,
        working_dir: Path,
        llm_api_key: str,
        llm_api_base: str,
        llm_model: str = "gpt-4o-mini",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
        project_id: str = "default",  # ← 添加 project_id
        # 其他参数保留兼容性
        **kwargs
    ):
        """初始化 LightRAG 图谱构建器"""
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.llm_model = llm_model
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.project_id = project_id  # ← 保存 project_id
        
        # 为 Leiden 社区检测添加别名
        self.uri = neo4j_uri
        self.user = neo4j_user
        self.password = neo4j_password
        self.database = kwargs.get('database', 'neo4j')  # 默认数据库名
        
        # 初始化组件
        self.extractor: Optional[GauzRagEntityExtractor] = None
        self.neo4j_store: Optional[Neo4jEntityStore] = None
        self.entity_mapper: Optional[GauzRagEntityMapper] = None
        self.initialized = False
    
    async def initialize(self):
        """初始化所有组件"""
        if self.initialized:
            return
        
        print(f"初始化 LightRAG 图谱构建器...")
        print(f"  - LLM: {self.llm_model}")
        print(f"  - Neo4j: {self.neo4j_uri}")
        print(f"  - 模式: Prompt提取 → Neo4j存储 → 倒排索引")
        
        # 初始化提取器
        self.extractor = GauzRagEntityExtractor(
            llm_api_key=self.llm_api_key,
            llm_api_base=self.llm_api_base,
            llm_model=self.llm_model,
            language="English"  # 统一使用英文输出
        )
        
        # 初始化 Neo4j 存储（带 project_id 和 database）
        self.neo4j_store = Neo4jEntityStore(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            project_id=self.project_id,
            database=self.database
        )
        
        # 创建 Neo4j 约束和索引（防止实体重复）
        await self.neo4j_store.create_constraints()
        
        # 初始化实体映射器
        self.entity_mapper = GauzRagEntityMapper(self.working_dir)
        
        self.initialized = True
        print(f"  ✓ 初始化完成")
    
    async def aclose(self):
        """关闭所有异步资源"""
        if hasattr(self, 'extractor') and self.extractor:
            await self.extractor.aclose()
        if hasattr(self, 'neo4j_store') and self.neo4j_store:
            await self.neo4j_store.close()
    
    async def insert_facts(
        self,
        facts: List[Dict[str, Any]],
        is_incremental: bool = True
    ) -> None:
        """
        处理 Facts：提取实体 → 存 Neo4j → 更新映射
        
        Args:
            facts: Facts 列表 [{'fact_id': int, 'content': str}, ...]
            is_incremental: 是否增量（Neo4j 天然支持）
        """
        if not self.initialized:
            await self.initialize()
        
        import time
        print(f"\n🚀 LightRAG 极速处理 {len(facts)} 条 Facts...")
        
        # ===== Phase 1: 并发提取所有Facts的实体 =====
        phase1_start = time.time()
        print(f"  [Phase 1] 并发提取实体...")
        
        async def extract_single_fact_entities(fact):
            """提取单个Fact的实体（异步）"""
            fact_id = fact.get("fact_id")
            content = fact.get("content", "")
            conversation_id = fact.get("conversation_id")  # 🔑 获取 conversation_id
            
            if not content:
                return None
            
            try:
                # 提取实体（已优化：不提取实体关系）
                entities = await self.extractor.extract_from_fact(content)
                
                # 标准化实体名称
                for entity in entities:
                    entity['name'] = self.extractor.normalize_entity_name(entity['name'])
                
                result = {
                    'fact_id': fact_id,
                    'content': content,
                    'entities': entities
                }
                
                # 🔑 如果有 conversation_id，也传递
                if conversation_id is not None:
                    result['conversation_id'] = conversation_id
                
                return result
            except Exception as e:
                print(f"  ✗ Fact {fact_id} 提取失败: {e}")
                return None
        
        # 🚀 MAP: 全并发提取
        extraction_results = await asyncio.gather(*[
            extract_single_fact_entities(fact) 
            for fact in facts
        ])
        
        # 过滤失败的结果
        extraction_results = [r for r in extraction_results if r is not None]
        
        phase1_time = time.time() - phase1_start
        total_entities = sum(len(r['entities']) for r in extraction_results)
        print(f"    ✓ 提取完成 [耗时: {phase1_time:.2f}秒]")
        print(f"    - 提取了 {total_entities} 个实体（含重复）")
        
        # ===== REDUCE: 内存聚合去重 =====
        phase2_start = time.time()
        print(f"  [Phase 2] 内存聚合去重...")
        
        # 1. 聚合所有唯一实体（按 name 去重）
        unique_entities = {}  # {entity_name: {type, description}}
        for result in extraction_results:
            for entity in result['entities']:
                name = entity['name']
                if name not in unique_entities:
                    unique_entities[name] = {
                        'type': entity['type'],
                        'description': entity['description']
                    }
                # 如果已存在，保留更详细的描述
                elif len(entity.get('description', '')) > len(unique_entities[name].get('description', '')):
                    unique_entities[name]['description'] = entity['description']
        
        print(f"    ✓ 去重完成: {len(unique_entities)} 个唯一实体")
        
        # ===== Phase 3: 批量写入Neo4j =====
        phase3_start = time.time()
        print(f"  [Phase 3] 批量写入Neo4j...")
        
        try:
            await self.neo4j_store.batch_store_facts_and_entities(
                extraction_results=extraction_results,
                unique_entities=unique_entities
            )
            print(f"    ✓ 批量写入完成 [耗时: {time.time() - phase3_start:.2f}秒]")
        except Exception as e:
            print(f"    ✗ 批量写入失败: {e}")
            raise
        
        phase2_time = time.time() - phase2_start
        
        # 构建fact_id到实体的映射（用于Phase 4内存注入）
        fact_entities_map = {
            r['fact_id']: r['entities'] 
            for r in extraction_results
        }
        
        store_results = [(r['fact_id'], [e['name'] for e in r['entities']]) for r in extraction_results]
        
        # ===== Phase 4: [已废弃] 更新Entity→Facts映射（实体已作为Fact属性存储）=====
        # print(f"  [Phase 4] 更新Entity→Facts映射...")
        # for fact_id, entity_names in store_results:
        #     self.entity_mapper.add_fact(fact_id, entity_names)
        # self.entity_mapper.save()
        print(f"  [Phase 4] 跳过Entity→Facts映射（实体已作为Fact属性存储）")
        
        # 打印统计
        total_time = phase1_time + phase2_time
        print(f"\n  ✓ 极速处理完成 [总耗时: {total_time:.2f}秒]")
        print(f"    - Phase 1: {phase1_time:.2f}秒, Phase 2: {phase2_time:.2f}秒")
        
        # 返回实体映射（供Phase 5使用）
        return fact_entities_map
    
    async def extract_fact_entities(self) -> Dict[int, List[str]]:
        """从 Neo4j 提取 Fact→Entity 映射（直接从Fact节点属性读取）"""
        if not self.initialized:
            await self.initialize()
        
        print(f"\n从 Neo4j 提取 Fact→Entity 映射...")
        
        async with self.neo4j_store.driver.session() as session:
            result = await session.run(
                """
                MATCH (f:Fact {project_id: $project_id})
                WHERE f.entities IS NOT NULL
                RETURN f.fact_id AS fact_id, f.entities AS entities
                """,
                project_id=self.project_id
            )
            
            fact_to_entities = {}
            async for record in result:
                fact_to_entities[record['fact_id']] = record['entities']
        
        print(f"  ✓ 提取完成: {len(fact_to_entities)} 个 Facts")
        if fact_to_entities:
            avg_entities = sum(len(e) for e in fact_to_entities.values()) / len(fact_to_entities)
            print(f"    - 平均每个 Fact 有 {avg_entities:.1f} 个实体")
        
        return fact_to_entities
    
    def get_entity_mapper(self) -> GauzRagEntityMapper:
        """获取实体映射器（用于构建 Facts 关系）"""
        if not self.entity_mapper:
            raise RuntimeError("请先调用 initialize()")
        return self.entity_mapper
    
    # ========== Leiden 社区检测功能 ==========
    
    def run_leiden_community_detection(
        self,
        resolution: float = 1.0,
        min_community_size: int = 3
    ) -> Dict[str, Any]:
        """
        执行 Leiden 社区检测（同步方法）
        
        核心优势：
        - 降低 Fact 关系匹配复杂度：O(n²) → O(k²)
        - 优化图可视化：避免密集实体图
        - 加速检索：基于社区过滤
        
        Args:
            resolution: Leiden 分辨率参数（越大社区越多越小）
            min_community_size: 最小社区大小（过滤小社区）
        
        Returns:
            社区检测结果和统计信息
        """
        print(f"\n{'='*60}")
        print(f"🔍 执行 Leiden 社区检测 (project: {self.project_id})")
        print(f"{'='*60}")
        
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            result = detector.detect_communities(
                project_id=self.project_id,
                resolution=resolution,
                min_community_size=min_community_size
            )
            
            print(f"\n✓ 社区检测完成！")
            print(f"  - 总实体数：{result['total_entities']}")
            print(f"  - 社区数：{result['total_communities']}")
            print(f"  - 平均社区大小：{result['total_entities'] / max(result['total_communities'], 1):.1f}")
            
            return result
        
        finally:
            detector.close()
    
    def rebuild_fact_relations_by_community(
        self,
        min_shared_entities: int = 2
    ) -> Dict[str, int]:
        """
        基于社区重建 Fact 关系（降低时间复杂度）
        
        Args:
            min_shared_entities: 最小共享实体数
        
        Returns:
            统计信息
        """
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            return detector.rebuild_fact_relations_by_community(
                project_id=self.project_id,
                min_shared_entities=min_shared_entities
            )
        finally:
            detector.close()
    
    def get_community_statistics(self) -> Dict[str, Any]:
        """获取社区统计信息"""
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            return detector.get_community_statistics(self.project_id)
        finally:
            detector.close()
    
    def list_communities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """列出所有社区"""
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            return detector.list_communities(self.project_id, limit=limit)
        finally:
            detector.close()
    
    def get_entity_community(self, entity_name: str) -> Optional[int]:
        """获取实体所属社区"""
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            return detector.get_entity_community(entity_name, self.project_id)
        finally:
            detector.close()
    
    def clear_communities(self) -> Dict[str, int]:
        """清除所有社区数据"""
        detector = LeidenCommunityDetector(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        try:
            return detector.clear_communities(self.project_id)
        finally:
            detector.close()
    
    async def close(self):
        """关闭所有连接"""
        if self.neo4j_store:
            await self.neo4j_store.close()

