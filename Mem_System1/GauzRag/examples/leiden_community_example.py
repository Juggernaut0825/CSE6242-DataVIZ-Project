"""
Leiden 社区检测使用示例
展示如何使用 Leiden 算法优化实体图和 Fact 关系匹配
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from GauzRag.lightrag_graph_builder import LightRAGGraphBuilder
from GauzRag.leiden_community_detector import LeidenCommunityDetector
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


async def main():
    """主函数"""
    
    # 配置信息
    PROJECT_ID = "test_leiden"
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Aa@123456")
    
    LLM_API_KEY = os.getenv("GAUZ_LLM_API_KEY")
    LLM_API_BASE = os.getenv("GAUZ_LLM_API_BASE")
    LLM_MODEL = os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini")
    
    print("="*80)
    print("Leiden 社区检测完整流程示例")
    print("="*80)
    
    # 步骤 1: 初始化 LightRAG Builder
    print("\n[步骤 1] 初始化 LightRAG Builder")
    builder = LightRAGGraphBuilder(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        project_id=PROJECT_ID,
        llm_api_key=LLM_API_KEY,
        llm_api_base=LLM_API_BASE,
        llm_model=LLM_MODEL
    )
    
    try:
        # 步骤 2: 假设已经有 Facts 和 Entities（这里跳过数据准备）
        print("\n[步骤 2] 检查现有数据")
        
        # 获取实体统计
        entity_stats = await builder.neo4j_store.get_entity_count(PROJECT_ID)
        print(f"  - 现有实体数：{entity_stats}")
        
        if entity_stats == 0:
            print("\n⚠️  没有实体数据！请先执行以下步骤：")
            print("  1. 运行 fact 提取")
            print("  2. 运行实体提取")
            print("  3. 构建实体-实体关系")
            return
        
        # 步骤 3: 执行 Leiden 社区检测
        print("\n[步骤 3] 执行 Leiden 社区检测")
        print("-" * 60)
        
        community_result = builder.run_leiden_community_detection(
            resolution=1.0,  # 分辨率参数（越大社区越多）
            min_community_size=3  # 最小社区大小
        )
        
        print(f"\n社区检测结果：")
        print(f"  - 总实体数：{community_result['total_entities']}")
        print(f"  - 总关系数：{community_result['total_relations']}")
        print(f"  - 检测到的社区数：{community_result['total_communities']}")
        
        print(f"\n前 5 个最大社区：")
        for i, community in enumerate(community_result['communities'][:5], 1):
            print(f"  {i}. 社区 {community['community_id']}: "
                  f"{community['size']} 个实体")
            print(f"     实体样例: {', '.join(community['entities'][:5])}")
        
        # 步骤 4: 查看社区统计
        print("\n[步骤 4] 社区统计信息")
        print("-" * 60)
        
        stats = builder.get_community_statistics()
        print(f"  - 总社区数：{stats['total_communities']}")
        print(f"  - 总实体数：{stats['total_entities']}")
        print(f"  - 平均社区大小：{stats['avg_community_size']}")
        print(f"  - 最大社区大小：{stats['largest_community_size']}")
        print(f"  - 最小社区大小：{stats['smallest_community_size']}")
        print(f"  - 未聚类实体数：{stats['unclustered_entities']}")
        
        # 步骤 5: 基于社区重新构建 Fact 关系
        print("\n[步骤 5] 基于社区重新构建 Fact 关系")
        print("-" * 60)
        print("这将大幅降低关系匹配的时间复杂度：")
        print("  - 全局模式：O(n²) - 所有 Facts 两两匹配")
        print("  - 社区模式：O(k²) - 只在同一社区内匹配（k << n）")
        
        relation_result = builder.rebuild_fact_relations_by_community(
            min_shared_entities=2  # 至少共享 2 个实体才建立关系
        )
        
        print(f"\n关系重建结果：")
        print(f"  - 处理的社区数：{relation_result['communities_processed']}")
        print(f"  - 创建的关系数：{relation_result['relations_created']}")
        
        # 步骤 6: 演示：基于社区的候选 Fact 查找
        print("\n[步骤 6] 演示：基于社区的候选查找")
        print("-" * 60)
        
        # 假设有一个新 Fact 包含以下实体
        test_entities = ["张三", "AI公司", "北京"]
        test_fact_id = 9999  # 假设的新 Fact ID
        
        detector = LeidenCommunityDetector(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        
        try:
            candidates = detector.get_candidate_facts_by_community(
                new_fact_id=test_fact_id,
                entities=test_entities,
                project_id=PROJECT_ID,
                max_candidates=20
            )
            
            print(f"\n实体 {test_entities} 的社区候选 Facts：")
            print(f"  - 找到 {len(candidates)} 个候选")
            print(f"  - 候选 IDs: {candidates[:10]}")  # 显示前 10 个
            
        finally:
            detector.close()
        
        # 步骤 7: 对比测试：全局 vs 社区优化
        print("\n[步骤 7] 性能对比：全局模式 vs 社区优化模式")
        print("-" * 60)
        
        import time
        
        # 全局模式（会很慢）
        print("\n测试 1: 全局模式构建关系")
        start_time = time.time()
        
        global_result = await builder.build_fact_relations(
            use_community_optimization=False,
            min_shared_entities=2
        )
        
        global_time = time.time() - start_time
        print(f"  - 耗时：{global_time:.2f} 秒")
        print(f"  - 创建关系数：{global_result['edges']}")
        
        # 社区优化模式（应该快很多）
        print("\n测试 2: 社区优化模式构建关系")
        
        # 先删除旧关系
        async with builder.driver.session() as session:
            await session.run("""
                MATCH (f1:Fact {project_id: $project_id})-[r:RELATED_TO]-(f2:Fact {project_id: $project_id})
                DELETE r
            """, project_id=PROJECT_ID)
        
        start_time = time.time()
        
        community_result = await builder.build_fact_relations(
            use_community_optimization=True,
            min_shared_entities=2
        )
        
        community_time = time.time() - start_time
        print(f"  - 耗时：{community_time:.2f} 秒")
        print(f"  - 创建关系数：{community_result['edges']}")
        
        # 对比结果
        print(f"\n性能提升：")
        if global_time > 0:
            speedup = global_time / max(community_time, 0.001)
            print(f"  - 加速比：{speedup:.2f}x")
            print(f"  - 时间节省：{global_time - community_time:.2f} 秒")
        
        print("\n" + "="*80)
        print("✓ 示例完成！")
        print("="*80)
        
        print("\n关键优势总结：")
        print("  1. 降低复杂度：O(n²) → O(k²)")
        print("  2. 避免密集图：社区内部密集，社区之间稀疏")
        print("  3. 加速检索：基于社区过滤候选")
        print("  4. 优化可视化：按社区着色，层次清晰")
        
    finally:
        await builder.close()


if __name__ == "__main__":
    asyncio.run(main())

