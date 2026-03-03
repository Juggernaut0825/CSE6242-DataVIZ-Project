"""
清理 Neo4j 中重复的实体节点
由于之前没有唯一性约束，同一个实体名可能被创建多次
"""
import asyncio
from neo4j import AsyncGraphDatabase
import os
from pathlib import Path
from dotenv import load_dotenv


async def cleanup_duplicate_entities(
    uri: str,
    user: str,
    password: str,
    project_id: str,
    database: str = "neo4j",
    dry_run: bool = True
):
    """
    清理重复的实体节点，保留最早创建的节点
    
    Args:
        uri: Neo4j连接URI
        user: 用户名
        password: 密码
        project_id: 项目ID
        database: 数据库名
        dry_run: 是否只检查不删除（默认True）
    """
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    try:
        async with driver.session(database=database) as session:
            # 1. 查找重复的实体（返回elementId列表）
            result = await session.run(
                """
                MATCH (e:Entity {project_id: $project_id})
                WITH e.name as entity_name, collect(elementId(e)) as entity_ids, count(e) as cnt
                WHERE cnt > 1
                RETURN entity_name, entity_ids, cnt
                ORDER BY cnt DESC
                """,
                project_id=project_id
            )
            
            duplicates = await result.data()
            
            if not duplicates:
                print(f"✓ 未发现重复实体（项目: {project_id}）")
                return
            
            print(f"\n发现 {len(duplicates)} 个重复的实体名:")
            print("="*80)
            
            total_duplicates = 0
            for record in duplicates:
                entity_name = record['entity_name']
                cnt = record['cnt']
                total_duplicates += (cnt - 1)
                print(f"  - {entity_name}: {cnt} 个节点")
            
            print(f"\n共需要清理 {total_duplicates} 个重复节点")
            print("="*80)
            
            if dry_run:
                print("\n[DRY RUN] 仅检查，未执行删除")
                print("如需删除，请运行: python GauzRag/cleanup_duplicate_entities.py --project-id <project_id> --delete")
                return
            
            # 2. 清理重复节点（保留第一个）
            print("\n开始清理...")
            cleaned_count = 0
            
            for record in duplicates:
                entity_name = record['entity_name']
                entity_ids = record['entity_ids']
                
                # 保留第一个ID，删除其余的
                keep_id = entity_ids[0]
                delete_ids = entity_ids[1:]
                
                print(f"\n处理实体: {entity_name}")
                print(f"  保留节点ID: {keep_id}")
                print(f"  删除 {len(delete_ids)} 个重复节点")
                
                # 逐个删除重复节点，并将其关系转移到保留的节点
                for dup_id in delete_ids:
                    # Step 1: 转移 Fact->Entity 关系
                    await session.run(
                        """
                        MATCH (dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        MATCH (keep:Entity)
                        WHERE elementId(keep) = $keep_id
                        MATCH (f:Fact)-[:HAS_ENTITY]->(dup)
                        MERGE (f)-[:HAS_ENTITY]->(keep)
                        """,
                        dup_id=dup_id,
                        keep_id=keep_id
                    )
                    
                    # Step 2: 转移 Entity->Entity 关系（出边）
                    await session.run(
                        """
                        MATCH (dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        MATCH (keep:Entity)
                        WHERE elementId(keep) = $keep_id
                        MATCH (dup)-[:RELATES_TO]->(target:Entity)
                        MERGE (keep)-[:RELATES_TO]->(target)
                        """,
                        dup_id=dup_id,
                        keep_id=keep_id
                    )
                    
                    # Step 3: 转移 Entity->Entity 关系（入边）
                    await session.run(
                        """
                        MATCH (dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        MATCH (keep:Entity)
                        WHERE elementId(keep) = $keep_id
                        MATCH (source:Entity)-[:RELATES_TO]->(dup)
                        MERGE (source)-[:RELATES_TO]->(keep)
                        """,
                        dup_id=dup_id,
                        keep_id=keep_id
                    )
                    
                    # Step 4: 删除重复节点（包括所有关系）
                    await session.run(
                        """
                        MATCH (dup:Entity)
                        WHERE elementId(dup) = $dup_id
                        DETACH DELETE dup
                        """,
                        dup_id=dup_id
                    )
                    cleaned_count += 1
                
                print(f"  ✓ 完成")
            
            print("\n" + "="*80)
            print(f"✓ 清理完成！共删除 {cleaned_count} 个重复节点")
    
    finally:
        await driver.close()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='清理Neo4j重复实体')
    parser.add_argument('--project-id', type=str, help='项目ID')
    parser.add_argument('--delete', action='store_true', help='执行删除（默认只检查）')
    args = parser.parse_args()
    
    # 加载环境变量
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # 获取配置
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    project_id = args.project_id or os.getenv("PROJECT_ID", "default")
    
    if not password:
        print("错误: 未找到 NEO4J_PASSWORD 环境变量")
        return
    
    print(f"Neo4j URI: {uri}")
    print(f"数据库: {database}")
    print(f"项目ID: {project_id}")
    print(f"模式: {'删除模式' if args.delete else '检查模式'}")
    print()
    
    await cleanup_duplicate_entities(
        uri=uri,
        user=user,
        password=password,
        project_id=project_id,
        database=database,
        dry_run=not args.delete
    )


if __name__ == "__main__":
    asyncio.run(main())

