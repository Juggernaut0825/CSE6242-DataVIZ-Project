"""
Leiden 社区检测模块
基于 Neo4j GDS 实现实体图的社区划分，优化 Fact 关系匹配性能
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import os
from collections import defaultdict


class LeidenCommunityDetector:
    """
    使用 Leiden 算法对实体图进行社区检测
    
    核心优势：
    - 降低关系匹配复杂度：O(n²) → O(k²)，k 是社区内节点数
    - 避免密集实体图：按社区分组，可视化更清晰
    - 加速检索：只在相关社区内搜索
    """
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = "neo4j"
    ):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j Bolt 连接地址
            user: 用户名
            password: 密码
            database: 数据库名
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "Aa@123456")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.database = database
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ========== 核心功能：Leiden 社区检测 ==========
    
    def detect_communities(
        self,
        project_id: str,
        graph_name: str = "entity_graph",
        resolution: float = 1.0,
        min_community_size: int = 3
    ) -> Dict[str, Any]:
        """
        对实体图执行 Leiden 社区检测
        
        流程：
        1. 从 Neo4j 投影实体图（Entity 节点 + RELATES_TO 关系）
        2. 执行 Leiden 算法
        3. 将社区 ID 写回 Entity 节点
        4. 创建 Community 节点并建立关系
        
        Args:
            project_id: 项目 ID（用于数据隔离）
            graph_name: 图投影名称
            resolution: 分辨率参数（越大社区越多越小）
            min_community_size: 最小社区大小（过滤小社区）
        
        Returns:
            {
                'total_entities': int,
                'total_communities': int,
                'communities': [
                    {
                        'community_id': int,
                        'size': int,
                        'entities': [str, ...]
                    }
                ]
            }
        """
        with self.driver.session(database=self.database) as session:
            # 步骤 1: 清理旧的图投影
            print(f"[1/5] 清理旧图投影: {graph_name}")
            self._drop_graph_if_exists(session, graph_name)
            
            # 步骤 2: 投影实体图
            print(f"[2/5] 投影实体图: {project_id}")
            entity_count, relation_count = self._project_entity_graph(
                session, graph_name, project_id
            )
            
            if entity_count == 0 or relation_count == 0:
                print("⚠️  没有实体或关系，跳过社区检测")
                self._drop_graph_if_exists(session, graph_name)
                return {
                    'total_entities': entity_count,
                    'total_communities': 0,
                    'communities': []
                }

            # 步骤 3: 执行 Leiden 算法
            print(f"[3/5] 执行 Leiden 算法 (resolution={resolution})")
            leiden_stats = self._run_leiden_algorithm(
                session, graph_name, resolution
            )
            
            # 步骤 4: 将社区 ID 写回 Entity 节点
            print(f"[4/5] 写回社区 ID 到 Entity 节点")
            self._write_community_ids(session, graph_name, project_id)
            
            # 步骤 5: 清理投影图
            print(f"[5/5] 清理图投影")
            self._drop_graph_if_exists(session, graph_name)
            self._drop_graph_if_exists(session, f"{graph_name}_full")
            
            # 步骤 6: 创建 Community 节点
            print(f"[6/6] 创建 Community 节点和关系")
            communities = self._create_community_nodes(
                session, project_id, min_community_size
            )
            
            result = {
                'total_entities': entity_count,
                'total_relations': relation_count,
                'total_communities': len(communities),
                'communities': communities,
                'leiden_stats': leiden_stats
            }
            
            print(f"✓ 社区检测完成：{entity_count} 个实体 → {len(communities)} 个社区")
            return result
    
    def _drop_graph_if_exists(self, session, graph_name: str):
        """删除已存在的图投影"""
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass  # 图不存在时忽略错误
    
    def _project_entity_graph(
        self,
        session,
        graph_name: str,
        project_id: str
    ) -> tuple[int, int]:
        """
        投影实体图到 GDS（无向图）

        使用 native projection + 临时标签，仅投影当前 project 的节点与关系，并设为 UNDIRECTED。
        """
        temp_label = f"ProjTemp_{project_id}"

        # 1) 给当前项目的节点打临时标签
        session.run(
            f"""
            MATCH (e:Entity {{project_id: $project_id}})
            SET e:`{temp_label}`
            """,
            project_id=project_id,
        )

        try:
            # 2) 投影：只包含临时标签节点，关系自动受限于节点集；关系设为 UNDIRECTED
            result = session.run(
                f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    ['{temp_label}'],
                    {{
                        RELATES_TO: {{
                            orientation: 'UNDIRECTED'
                        }}
                    }}
                )
                YIELD nodeCount, relationshipCount
                RETURN nodeCount, relationshipCount
                """
            ).single()

            return result["nodeCount"], result["relationshipCount"]

        finally:
            # 3) 清理临时标签
            session.run(
                f"""
                MATCH (e:`{temp_label}`)
                REMOVE e:`{temp_label}`
                """,
                project_id=project_id,
            )
    
    def _run_leiden_algorithm(
        self,
        session,
        graph_name: str,
        resolution: float = 1.0
    ) -> Dict[str, Any]:
        """
        执行 Leiden 社区检测算法
        
        参数：
        - resolution: 分辨率（1.0 为标准值，越大社区越多）
        """
        # 使用默认配置（依赖投影时的 UNDIRECTED 设置），避免版本不支持的参数
        result = session.run(f"""
            CALL gds.leiden.stream('{graph_name}', {{
                gamma: {resolution},
                includeIntermediateCommunities: false
            }})
            YIELD nodeId, communityId, intermediateCommunityIds
            RETURN 
                count(DISTINCT communityId) AS total_communities,
                count(nodeId) AS total_nodes,
                min(communityId) AS min_community_id,
                max(communityId) AS max_community_id
        """).single()
        
        return dict(result)
    
    def _write_community_ids(
        self,
        session,
        graph_name: str,
        project_id: str
    ):
        """
        将 Leiden 算法结果写回 Entity 节点的 community_id 属性
        """
        session.run(f"""
            CALL gds.leiden.write('{graph_name}', {{
                writeProperty: 'community_id'
            }})
            YIELD communityCount, nodePropertiesWritten
        """)
    
    def _create_community_nodes(
        self,
        session,
        project_id: str,
        min_community_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        创建 Community 节点，并建立 (Entity)-[:BELONGS_TO]->(Community) 关系
        
        Returns:
            社区列表，每个社区包含：
            {
                'community_id': int,
                'size': int,
                'entities': [str, ...]
            }
        """
        # 1. 获取所有社区及其实体
        result = session.run("""
            MATCH (e:Entity {project_id: $project_id})
            WHERE e.community_id IS NOT NULL
            WITH e.community_id AS community_id, collect(e.name) AS entities
            WHERE size(entities) >= $min_size
            RETURN community_id, entities, size(entities) AS size
            ORDER BY size DESC
        """, project_id=project_id, min_size=min_community_size).data()
        
        communities = []
        
        # 2. 为每个社区创建节点
        for record in result:
            community_id = record['community_id']
            entities = record['entities']
            size = record['size']
            
            # 创建 Community 节点
            session.run("""
                MERGE (c:Community {community_id: $community_id, project_id: $project_id})
                SET c.size = $size,
                    c.entity_count = $size,
                    c.updated_at = datetime()
            """, community_id=community_id, project_id=project_id, size=size)
            
            # 建立关系
            session.run("""
                MATCH (e:Entity {project_id: $project_id})
                WHERE e.community_id = $community_id
                MATCH (c:Community {community_id: $community_id, project_id: $project_id})
                MERGE (e)-[:BELONGS_TO]->(c)
            """, community_id=community_id, project_id=project_id)
            
            communities.append({
                'community_id': community_id,
                'size': size,
                'entities': entities
            })
        
        return communities
    
    # ========== 社区查询功能 ==========
    
    def get_entity_community(
        self,
        entity_name: str,
        project_id: str
    ) -> Optional[int]:
        """
        获取实体所属的社区 ID
        
        Args:
            entity_name: 实体名称
            project_id: 项目 ID
        
        Returns:
            community_id 或 None
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity {name: $name, project_id: $project_id})
                RETURN e.community_id AS community_id
            """, name=entity_name, project_id=project_id).single()
            
            if result:
                return result['community_id']
            return None
    
    def get_community_entities(
        self,
        community_id: int,
        project_id: str
    ) -> List[str]:
        """
        获取社区内的所有实体
        
        Args:
            community_id: 社区 ID
            project_id: 项目 ID
        
        Returns:
            实体名称列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity {community_id: $community_id, project_id: $project_id})
                RETURN e.name AS name
            """, community_id=community_id, project_id=project_id).data()
            
            return [record['name'] for record in result]
    
    def get_all_communities(
        self,
        project_id: str,
        min_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取所有社区信息
        
        Args:
            project_id: 项目 ID
            min_size: 最小社区大小
        
        Returns:
            社区列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Community {project_id: $project_id})
                WHERE c.size >= $min_size
                OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(c)
                RETURN 
                    c.community_id AS community_id,
                    c.size AS size,
                    collect(e.name) AS entities
                ORDER BY c.size DESC
            """, project_id=project_id, min_size=min_size).data()
            
            return result
    
    def get_community_statistics(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        获取社区统计信息
        
        Returns:
            {
                'total_communities': int,
                'total_entities': int,
                'avg_community_size': float,
                'largest_community_size': int,
                'smallest_community_size': int,
                'unclustered_entities': int
            }
        """
        with self.driver.session(database=self.database) as session:
            # 社区统计
            community_stats = session.run("""
                MATCH (c:Community {project_id: $project_id})
                RETURN 
                    count(c) AS total_communities,
                    avg(c.size) AS avg_size,
                    max(c.size) AS max_size,
                    min(c.size) AS min_size
            """, project_id=project_id).single()
            
            # 未聚类实体
            unclustered = session.run("""
                MATCH (e:Entity {project_id: $project_id})
                WHERE e.community_id IS NULL
                RETURN count(e) AS unclustered_count
            """, project_id=project_id).single()
            
            # 总实体数
            total_entities = session.run("""
                MATCH (e:Entity {project_id: $project_id})
                RETURN count(e) AS total
            """, project_id=project_id).single()
            
            return {
                'total_communities': community_stats['total_communities'] or 0,
                'total_entities': total_entities['total'] or 0,
                'avg_community_size': round(community_stats['avg_size'] or 0, 2),
                'largest_community_size': community_stats['max_size'] or 0,
                'smallest_community_size': community_stats['min_size'] or 0,
                'unclustered_entities': unclustered['unclustered_count'] or 0
            }
    
    # ========== 基于社区的 Fact 关系匹配优化 ==========
    
    def get_candidate_facts_by_community(
        self,
        new_fact_id: int,
        entities: List[str],
        project_id: str,
        max_candidates: int = 100
    ) -> List[int]:
        """
        基于社区获取候选 Facts（用于关系匹配）
        
        核心优化：
        - 只返回与新 Fact 共享社区的 Facts
        - 时间复杂度：O(同一社区内的 Facts 数量)
        
        Args:
            new_fact_id: 新 Fact 的 ID
            entities: 新 Fact 包含的实体列表
            project_id: 项目 ID
            max_candidates: 最大候选数
        
        Returns:
            候选 Fact IDs 列表
        """
        with self.driver.session(database=self.database) as session:
            # 1. 获取新 Fact 的实体所属社区
            result = session.run("""
                MATCH (e:Entity {project_id: $project_id})
                WHERE e.name IN $entities AND e.community_id IS NOT NULL
                RETURN collect(DISTINCT e.community_id) AS community_ids
            """, entities=entities, project_id=project_id).single()
            
            community_ids = result['community_ids'] if result else []
            
            if not community_ids:
                # 如果没有社区信息，回退到全局搜索（但限制数量）
                print(f"⚠️  Fact {new_fact_id} 的实体未分配社区，使用全局搜索")
                result = session.run("""
                    MATCH (f:Fact {project_id: $project_id})
                    WHERE f.fact_id <> $new_fact_id
                    RETURN f.fact_id AS fact_id
                    LIMIT $max_candidates
                """, new_fact_id=new_fact_id, project_id=project_id, 
                     max_candidates=max_candidates).data()
                
                return [r['fact_id'] for r in result]
            
            # 2. 获取同一社区内的其他 Facts
            result = session.run("""
                MATCH (e:Entity {project_id: $project_id})
                WHERE e.community_id IN $community_ids
                MATCH (f:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e)
                WHERE f.fact_id <> $new_fact_id
                RETURN DISTINCT f.fact_id AS fact_id
                LIMIT $max_candidates
            """, community_ids=community_ids, new_fact_id=new_fact_id,
                 project_id=project_id, max_candidates=max_candidates).data()
            
            candidate_ids = [r['fact_id'] for r in result]
            
            print(f"✓ Fact {new_fact_id} 基于社区 {community_ids} 找到 {len(candidate_ids)} 个候选")
            return candidate_ids
    
    def rebuild_fact_relations_by_community(
        self,
        project_id: str,
        min_shared_entities: int = 2
    ) -> Dict[str, int]:
        """
        基于社区重新构建所有 Fact 关系
        
        只在同一社区内的 Facts 之间建立 RELATED_TO 关系
        
        Args:
            project_id: 项目 ID
            min_shared_entities: 最小共享实体数
        
        Returns:
            统计信息：{'relations_created': int, 'communities_processed': int}
        """
        with self.driver.session(database=self.database) as session:
            # 1. 删除旧的关系
            session.run("""
                MATCH (f1:Fact {project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                DELETE r
            """, project_id=project_id)
            
            # 2. 获取所有社区
            communities = session.run("""
                MATCH (c:Community {project_id: $project_id})
                RETURN c.community_id AS community_id, c.size AS size
                ORDER BY c.size DESC
            """, project_id=project_id).data()
            
            total_relations = 0
            
            # 3. 对每个社区内的 Facts 构建关系
            for community in communities:
                community_id = community['community_id']
                
                # 获取社区内的 Facts 及其实体
                facts_in_community = session.run("""
                    MATCH (f:Fact {project_id: $project_id})-[:HAS_ENTITY]->(e:Entity {community_id: $community_id})
                    WITH f, collect(e.name) AS entities
                    RETURN f.fact_id AS fact_id, entities
                """, project_id=project_id, community_id=community_id).data()
                
                # 构建倒排索引
                entity_to_facts = defaultdict(set)
                fact_entities = {}
                
                for record in facts_in_community:
                    fact_id = record['fact_id']
                    entities = record['entities']
                    fact_entities[fact_id] = set(entities)
                    
                    for entity in entities:
                        entity_to_facts[entity].add(fact_id)
                
                # 计算共享实体并创建关系
                for fact_id, entities in fact_entities.items():
                    candidates = set()
                    for entity in entities:
                        candidates.update(entity_to_facts[entity])
                    
                    candidates.discard(fact_id)  # 排除自己
                    
                    for candidate_id in candidates:
                        shared = entities & fact_entities[candidate_id]
                        
                        if len(shared) >= min_shared_entities:
                            # 创建关系（避免重复）
                            if fact_id < candidate_id:
                                session.run("""
                                    MATCH (f1:Fact {fact_id: $fact_id, project_id: $project_id})
                                    MATCH (f2:Fact {fact_id: $candidate_id, project_id: $project_id})
                                    MERGE (f1)-[r:RELATED_TO]-(f2)
                                    SET r.shared_entities = $shared_entities,
                                        r.weight = $weight
                                """, fact_id=fact_id, candidate_id=candidate_id,
                                     project_id=project_id,
                                     shared_entities=list(shared),
                                     weight=len(shared))
                                
                                total_relations += 1
                
                print(f"  社区 {community_id}: 创建了 {total_relations} 个关系")
            
            return {
                'relations_created': total_relations,
                'communities_processed': len(communities)
            }

