"""
Neo4j 图数据库存储模块
替代 JSON 文件，提供高性能图查询
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os
from datetime import datetime


class Neo4jGraphStore:
    """Neo4j 图存储管理器"""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = "neo4j",
        project_id: str = "default"
    ):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j Bolt 连接地址 (默认从环境变量 NEO4J_URI 读取)
            user: 用户名 (默认从环境变量 NEO4J_USER 读取)
            password: 密码 (默认从环境变量 NEO4J_PASSWORD 读取)
            database: 数据库名
            project_id: 项目ID（用于多项目隔离）
        """
        # 从环境变量读取配置（如果未提供）
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "Aa@123456")
        self.project_id = project_id
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.database = database
        
        # 创建索引和约束
        self._create_constraints()
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _create_constraints(self):
        """创建 Fact 节点约束和索引"""
        with self.driver.session(database=self.database) as session:
            # Fact 节点唯一性约束
            try:
                session.run("""
                    CREATE CONSTRAINT fact_id_unique IF NOT EXISTS
                    FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE
                """)
            except Exception as e:
                print(f"  ⚠️ Fact 约束创建跳过: {e}")
            
            # Fact 内容全文索引
            try:
                session.run("""
                    CREATE INDEX fact_content_fulltext IF NOT EXISTS
                    FOR (f:Fact) ON (f.content)
                """)
            except Exception as e:
                print(f"  ⚠️ Fact 内容索引创建跳过: {e}")
            
            # Fact 时间索引（用于时间范围查询）
            try:
                session.run("""
                    CREATE INDEX fact_created_at IF NOT EXISTS
                    FOR (f:Fact) ON (f.created_at)
                """)
            except Exception as e:
                print(f"  ⚠️ Fact 时间索引创建跳过: {e}")
            
            print("✓ Neo4j 约束和索引初始化完成")
    
    # ========== Fact 节点操作 ==========
    
    def add_fact_node(
        self,
        fact_id: int,
        content: str,
        entities: List[str] = None,
        communities: List[int] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        添加或更新 Fact 节点
        
        Args:
            fact_id: Fact ID
            content: Fact 内容
            entities: 实体列表
            communities: 社区 ID 列表
            metadata: 额外元数据
        """
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._create_fact,
                fact_id,
                content,
                entities or [],
                communities or [],
                metadata or {},
                self.project_id
            )
    
    @staticmethod
    def _create_fact(tx, fact_id, content, entities, communities, metadata, project_id):
        """事务：创建 Fact 节点"""
        query = """
        MERGE (f:Fact {fact_id: $fact_id, project_id: $project_id})
        SET f.content = $content,
            f.entities = $entities,
            f.communities = $communities,
            f.created_at = datetime($created_at),
            f.updated_at = datetime()
        """
        
        # 添加额外的 metadata 字段
        for key, value in metadata.items():
            query += f", f.{key} = ${key}"
        
        params = {
            "fact_id": fact_id,
            "project_id": project_id,
            "content": content,
            "entities": entities,
            "communities": communities,
            "created_at": metadata.get('created_at', datetime.now().isoformat()),
            **metadata
        }
        
        tx.run(query, **params)
        
        # 创建 Fact → Entity 关系（兼容LightRAG的 project_id 模式）
        for entity in entities:
            tx.run("""
                MERGE (e:Entity {name: $entity, project_id: $project_id})
                MERGE (f:Fact {fact_id: $fact_id, project_id: $project_id})
                MERGE (f)-[:HAS_ENTITY]->(e)
            """, entity=entity, fact_id=fact_id, project_id=project_id)
        
        # 创建 Fact → Community 关系
        for community_id in communities:
            tx.run("""
                MERGE (c:Community {community_id: $community_id})
                MERGE (f:Fact {fact_id: $fact_id, project_id: $project_id})
                MERGE (f)-[:BELONGS_TO]->(c)
            """, community_id=community_id, fact_id=fact_id, project_id=project_id)
    
    def add_fact_relation(
        self,
        source_fact_id: int,
        target_fact_id: int,
        relation_type: str,
        confidence: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        添加 Fact 之间的关系
        
        Args:
            source_fact_id: 源 Fact ID
            target_fact_id: 目标 Fact ID
            relation_type: 关系类型 (causality/temporal/thematic/contrast/elaboration)
            confidence: 置信度 (0-1)
            metadata: 额外元数据
        """
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._create_relation,
                source_fact_id,
                target_fact_id,
                relation_type,
                confidence,
                metadata or {},
                self.project_id
            )
    
    @staticmethod
    def _create_relation(tx, source_id, target_id, rel_type, confidence, metadata, project_id):
        """事务：创建 Fact 关系"""
        # 使用动态关系类型
        rel_type_upper = rel_type.upper()
        
        query = f"""
        MATCH (source:Fact {{fact_id: $source_id, project_id: $project_id}})
        MATCH (target:Fact {{fact_id: $target_id, project_id: $project_id}})
        MERGE (source)-[r:{rel_type_upper}]->(target)
        SET r.confidence = $confidence,
            r.created_at = datetime(),
            r.relation_type = $rel_type
        """
        
        # 添加额外的 metadata
        for key, value in metadata.items():
            query += f", r.{key} = ${key}"
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "project_id": project_id,
            "confidence": confidence,
            "rel_type": rel_type,
            **metadata
        }
        
        tx.run(query, **params)
    
    # ========== 批量导入 ==========
    
    def import_from_json(self, json_path: Path) -> Dict[str, int]:
        """
        从现有 JSON 文件导入到 Neo4j
        
        Args:
            json_path: fact_relations.json 路径
        
        Returns:
            {"nodes": count, "edges": count}
        """
        print(f"\n开始从 {json_path} 导入到 Neo4j...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        nodes_count = 0
        edges_count = 0
        
        # 导入节点
        print(f"  - 导入 {len(graph_data['nodes'])} 个节点...")
        for node_id, node_data in graph_data['nodes'].items():
            self.add_fact_node(
                fact_id=node_data['fact_id'],
                content=node_data['content'],
                entities=node_data.get('entities', []),
                communities=node_data.get('communities', []),
                metadata={
                    'created_at': node_data.get('created_at', datetime.now().isoformat())
                }
            )
            nodes_count += 1
        
        # 导入边
        print(f"  - 导入 {len(graph_data['edges'])} 条关系...")
        for edge in graph_data['edges']:
            # 解析 source/target（可能是 "fact_123" 格式）
            source_id = int(edge['source'].replace('fact_', ''))
            target_id = int(edge['target'].replace('fact_', ''))
            
            self.add_fact_relation(
                source_fact_id=source_id,
                target_fact_id=target_id,
                relation_type=edge['relation_type'],
                confidence=edge['confidence']
            )
            edges_count += 1
        
        print(f"✓ 导入完成: {nodes_count} 个节点, {edges_count} 条关系")
        
        return {"nodes": nodes_count, "edges": edges_count}
    
    def export_to_json(self, output_path: Path) -> None:
        """
        从 Neo4j 导出到 JSON（向后兼容）
        
        Args:
            output_path: 输出 JSON 路径
        """
        with self.driver.session(database=self.database) as session:
            # 获取所有节点
            nodes_result = session.run("""
                MATCH (f:Fact)
                RETURN f.fact_id AS fact_id,
                       f.content AS content,
                       f.entities AS entities,
                       f.communities AS communities,
                       f.created_at AS created_at
            """)
            
            nodes = {}
            for record in nodes_result:
                fact_id = str(record['fact_id'])
                nodes[fact_id] = {
                    'fact_id': record['fact_id'],
                    'content': record['content'],
                    'entities': record['entities'] or [],
                    'communities': record['communities'] or [],
                    'created_at': str(record['created_at'])
                }
            
            # 获取所有关系（无向边查询，避免重复）
            edges_result = session.run("""
                MATCH (source:Fact)-[r]-(target:Fact)
                WHERE id(source) < id(target)
                RETURN source.fact_id AS source,
                       target.fact_id AS target,
                       type(r) AS rel_type,
                       r.confidence AS confidence
            """)
            
            edges = []
            for record in edges_result:
                edges.append({
                    'source': f"fact_{record['source']}",
                    'target': f"fact_{record['target']}",
                    'relation_type': record['rel_type'].lower(),
                    'confidence': record['confidence']
                })
            
            # 保存
            graph_data = {
                'nodes': nodes,
                'edges': edges,
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_nodes': len(nodes),
                    'total_edges': len(edges)
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 导出到 {output_path}")
    
    # ========== 高级查询 ==========
    
    def find_related_facts(
        self,
        fact_id: int,
        max_hops: int = 2,
        min_confidence: float = 0.5,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查找相关 Facts（多跳查询）
        
        Args:
            fact_id: 起始 Fact ID
            max_hops: 最大跳数
            min_confidence: 最小置信度
            start_time: 开始时间（ISO 格式），例如 "2024-01-01T00:00:00"
            end_time: 结束时间（ISO 格式），例如 "2024-12-31T23:59:59"
        
        Returns:
            相关 Facts 列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH path = (start:Fact {fact_id: $fact_id, project_id: $project_id})-[r*1..$max_hops]-(related:Fact {project_id: $project_id})
                WHERE all(rel in relationships(path) WHERE rel.confidence >= $min_confidence)
                  AND ($start_time IS NULL OR related.created_at >= datetime($start_time))
                  AND ($end_time IS NULL OR related.created_at <= datetime($end_time))
                RETURN DISTINCT related.fact_id AS fact_id,
                       related.content AS content,
                       related.created_at AS created_at,
                       length(path) AS distance,
                       [rel in relationships(path) | type(rel)] AS path_types
                ORDER BY distance ASC, related.created_at DESC
                LIMIT 20
            """, 
                fact_id=fact_id,
                project_id=self.project_id,
                max_hops=max_hops, 
                min_confidence=min_confidence,
                start_time=start_time,
                end_time=end_time
            )
            
            return [dict(record) for record in result]
    
    def find_facts_by_entity(
        self,
        entity_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        查找包含特定实体的所有 Facts
        
        Args:
            entity_name: 实体名称
            start_time: 开始时间（ISO 格式）
            end_time: 结束时间（ISO 格式）
            limit: 返回数量限制
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (f:Fact)-[:HAS_ENTITY]->(e:Entity {name: $entity})
                WHERE ($start_time IS NULL OR f.created_at >= datetime($start_time))
                  AND ($end_time IS NULL OR f.created_at <= datetime($end_time))
                RETURN f.fact_id AS fact_id,
                       f.content AS content,
                       f.entities AS entities,
                       f.created_at AS created_at
                ORDER BY f.created_at DESC
                LIMIT $limit
            """, 
                entity=entity_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            return [dict(record) for record in result]
    
    def find_fact_community(
        self,
        fact_id: int,
        min_shared_entities: int = 2,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        基于共享实体查找 Fact 社区
        
        Args:
            fact_id: Fact ID
            min_shared_entities: 最小共享实体数
            start_time: 开始时间（ISO 格式）
            end_time: 结束时间（ISO 格式）
        
        Returns:
            社区成员 Facts
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (f1:Fact {fact_id: $fact_id, project_id: $project_id})-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(f2:Fact {project_id: $project_id})
                WHERE f1 <> f2
                  AND ($start_time IS NULL OR f2.created_at >= datetime($start_time))
                  AND ($end_time IS NULL OR f2.created_at <= datetime($end_time))
                WITH f2, count(DISTINCT e) AS shared_count
                WHERE shared_count >= $min_shared
                RETURN f2.fact_id AS fact_id,
                       f2.content AS content,
                       f2.created_at AS created_at,
                       shared_count
                ORDER BY shared_count DESC, f2.created_at DESC
                LIMIT 10
            """, 
                fact_id=fact_id, 
                min_shared=min_shared_entities,
                start_time=start_time,
                end_time=end_time
            )
            
            return [dict(record) for record in result]
    
    def find_facts_by_timerange(
        self,
        start_time: str,
        end_time: str,
        entity_filter: Optional[str] = None,
        limit: int = 100,
        order_by: str = "DESC"
    ) -> List[Dict[str, Any]]:
        """
        按时间范围查询 Facts（核心时间维度检索方法）
        
        Args:
            start_time: 开始时间（ISO 格式），如 "2024-01-01T00:00:00"
            end_time: 结束时间（ISO 格式），如 "2024-12-31T23:59:59"
            entity_filter: 可选的实体过滤（只返回包含此实体的 facts）
            limit: 返回数量限制
            order_by: 排序方式 "ASC" 或 "DESC"（默认最新在前）
        
        Returns:
            Facts 列表，包含 fact_id, content, entities, created_at
        
        示例:
            # 查询最近7天的所有 facts
            store.find_facts_by_timerange(
                start_time=(datetime.now() - timedelta(days=7)).isoformat(),
                end_time=datetime.now().isoformat()
            )
            
            # 查询某个时间段内提到"GPT-4"的 facts
            store.find_facts_by_timerange(
                start_time="2024-01-01T00:00:00",
                end_time="2024-06-30T23:59:59",
                entity_filter="GPT-4"
            )
        """
        with self.driver.session(database=self.database) as session:
            # 构建查询
            if entity_filter:
                query = """
                    MATCH (f:Fact)-[:HAS_ENTITY]->(e:Entity {name: $entity})
                    WHERE f.created_at >= datetime($start_time)
                      AND f.created_at <= datetime($end_time)
                    RETURN f.fact_id AS fact_id,
                           f.content AS content,
                           f.entities AS entities,
                           f.communities AS communities,
                           f.created_at AS created_at
                    ORDER BY f.created_at {order}
                    LIMIT $limit
                """.format(order=order_by)
                params = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "entity": entity_filter,
                    "limit": limit
                }
            else:
                query = """
                    MATCH (f:Fact)
                    WHERE f.created_at >= datetime($start_time)
                      AND f.created_at <= datetime($end_time)
                    RETURN f.fact_id AS fact_id,
                           f.content AS content,
                           f.entities AS entities,
                           f.communities AS communities,
                           f.created_at AS created_at
                    ORDER BY f.created_at {order}
                    LIMIT $limit
                """.format(order=order_by)
                params = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": limit
                }
            
            result = session.run(query, **params)
            return [dict(record) for record in result]
    
    def get_time_distribution(
        self,
        granularity: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        获取 Facts 的时间分布统计
        
        Args:
            granularity: 时间粒度 "hour", "day", "week", "month"
        
        Returns:
            时间分布列表 [{"time_bucket": "2024-01-01", "count": 10}, ...]
        """
        with self.driver.session(database=self.database) as session:
            # 根据粒度选择截断函数
            if granularity == "hour":
                truncate = "datetime.truncate('hour', f.created_at)"
            elif granularity == "day":
                truncate = "date(f.created_at)"
            elif granularity == "week":
                truncate = "date.truncate('week', f.created_at)"
            elif granularity == "month":
                truncate = "date.truncate('month', f.created_at)"
            else:
                truncate = "date(f.created_at)"
            
            query = f"""
                MATCH (f:Fact)
                WHERE f.created_at IS NOT NULL
                WITH {truncate} AS time_bucket
                RETURN toString(time_bucket) AS time_bucket, count(*) AS count
                ORDER BY time_bucket DESC
            """
            
            result = session.run(query)
            return [dict(record) for record in result]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (f:Fact)
                OPTIONAL MATCH (f)-[r]-()
                RETURN count(DISTINCT f) AS total_facts,
                       count(DISTINCT r) AS total_relations,
                       avg(size((f)-[]->())) AS avg_out_degree,
                       min(f.created_at) AS earliest_fact,
                       max(f.created_at) AS latest_fact
            """)
            
            record = result.single()
            
            return {
                'total_facts': record['total_facts'],
                'total_relations': record['total_relations'],
                'avg_out_degree': record['avg_out_degree'],
                'earliest_fact': str(record['earliest_fact']) if record['earliest_fact'] else None,
                'latest_fact': str(record['latest_fact']) if record['latest_fact'] else None
            }
    
    def run_community_detection(
        self,
        algorithm: str = "louvain",
        write_property: str = "community_id"
    ) -> Dict[str, int]:
        """
        使用 Neo4j GDS 运行社区检测
        
        Args:
            algorithm: "louvain" | "leiden" | "label_propagation"
            write_property: 将社区 ID 写入节点的属性名
        
        Returns:
            {"community_count": int}
        
        注意：需要安装 Neo4j Graph Data Science (GDS) 插件
        """
        with self.driver.session(database=self.database) as session:
            # 1. 创建图投影（改用 Native Projection，避免 id() 弃用警告）
            session.run("""
                CALL gds.graph.project(
                    'fact-graph',
                    'Fact',
                    {
                        RELATIONSHIP: {
                            orientation: 'UNDIRECTED',
                            properties: 'confidence'
                        }
                    }
                )
            """)
            
            # 2. 运行算法
            if algorithm == "louvain":
                result = session.run(f"""
                    CALL gds.louvain.write('fact-graph', {{
                        writeProperty: '{write_property}',
                        relationshipWeightProperty: 'confidence'
                    }})
                    YIELD communityCount, modularity
                    RETURN communityCount, modularity
                """)
            elif algorithm == "leiden":
                result = session.run(f"""
                    CALL gds.leiden.write('fact-graph', {{
                        writeProperty: '{write_property}',
                        relationshipWeightProperty: 'confidence'
                    }})
                    YIELD communityCount, modularity
                    RETURN communityCount, modularity
                """)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            record = result.single()
            
            # 3. 清理图投影
            session.run("CALL gds.graph.drop('fact-graph')")
            
            return {
                'community_count': record['communityCount'],
                'modularity': record['modularity']
            }


# ========== 便捷函数 ==========

def migrate_json_to_neo4j(
    json_path: Path,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "your_password_here"
) -> None:
    """
    一键迁移：JSON → Neo4j
    
    使用示例:
    ```python
    from GauzRag.neo4j_storage import migrate_json_to_neo4j
    
    migrate_json_to_neo4j(
        json_path=Path("output/test_project/fact_relations.json"),
        neo4j_password="your_actual_password"
    )
    ```
    """
    with Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_password) as store:
        stats = store.import_from_json(json_path)
        print(f"\n✅ 迁移完成!")
        print(f"  - 节点: {stats['nodes']}")
        print(f"  - 关系: {stats['edges']}")
        
        # 显示统计信息
        graph_stats = store.get_graph_statistics()
        print(f"\n图统计:")
        print(f"  - 总 Facts: {graph_stats['total_facts']}")
        print(f"  - 总关系: {graph_stats['total_relations']}")
        print(f"  - 平均出度: {graph_stats['avg_out_degree']:.2f}")

