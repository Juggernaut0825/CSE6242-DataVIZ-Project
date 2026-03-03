"""
时间维度检索示例

演示如何在 GauzRag 中使用时间维度进行知识图谱检索
"""
from pathlib import Path
from datetime import datetime, timedelta
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from GauzRag.config import GauzRagConfig
from GauzRag.pipeline import GauzRagPipeline
from GauzRag.neo4j_storage import Neo4jGraphStore


def example_1_basic_time_search():
    """示例 1: 基本的时间范围检索"""
    print("\n" + "="*80)
    print("示例 1: 基本的时间范围检索")
    print("="*80)
    
    # 初始化配置和 Pipeline
    config = GauzRagConfig.from_env(Path(".env"))
    pipeline = GauzRagPipeline(config, project_id="demo")
    
    # 查询最近7天的记忆
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    print(f"\n查询时间范围: {start_time.date()} ~ {end_time.date()}")
    
    results = pipeline.search_with_time_dimension(
        query="最近讨论了什么？",
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        top_k=10
    )
    
    print(f"\n结果统计:")
    print(f"  - 图检索: {len(results['graph_results'])} 条")
    print(f"  - 向量检索: {len(results['vector_results'])} 条")
    print(f"  - 合并后: {len(results['merged_results'])} 条")
    
    # 显示前5条结果
    print(f"\n前5条结果:")
    for i, fact in enumerate(results['merged_results'][:5], 1):
        print(f"\n{i}. Fact ID: {fact['fact_id']}")
        print(f"   内容: {fact['content'][:100]}...")
        print(f"   时间: {fact['created_at']}")
        print(f"   来源: {fact['source']}")
        if fact['score'] > 0:
            print(f"   相似度: {fact['score']:.3f}")


def example_2_entity_filter():
    """示例 2: 带实体过滤的时间检索"""
    print("\n" + "="*80)
    print("示例 2: 带实体过滤的时间检索")
    print("="*80)
    
    config = GauzRagConfig.from_env(Path(".env"))
    pipeline = GauzRagPipeline(config, project_id="demo")
    
    # 查询2024年上半年关于"GPT-4"的讨论
    results = pipeline.search_with_time_dimension(
        query="GPT-4 的最新功能和应用",
        start_time="2024-01-01T00:00:00",
        end_time="2024-06-30T23:59:59",
        top_k=20,
        entity_filter="GPT-4"  # 只返回提到 GPT-4 的 facts
    )
    
    print(f"\n找到 {len(results['merged_results'])} 条关于 GPT-4 的记录")
    
    # 按月份统计
    monthly_counts = {}
    for fact in results['merged_results']:
        if 'created_at' in fact and fact['created_at']:
            month = fact['created_at'][:7]  # 提取 "YYYY-MM"
            monthly_counts[month] = monthly_counts.get(month, 0) + 1
    
    print(f"\n月度分布:")
    for month in sorted(monthly_counts.keys()):
        print(f"  {month}: {monthly_counts[month]} 条")


def example_3_neo4j_only():
    """示例 3: 仅使用 Neo4j 图检索"""
    print("\n" + "="*80)
    print("示例 3: 仅使用 Neo4j 图检索（快速精确查询）")
    print("="*80)
    
    # 直接使用 Neo4j 存储
    with Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="Aa@123456"
    ) as store:
        # 查询特定日期的所有 facts
        results = store.find_facts_by_timerange(
            start_time="2024-12-01T00:00:00",
            end_time="2024-12-31T23:59:59",
            limit=50,
            order_by="DESC"  # 最新的在前
        )
        
        print(f"\n找到 {len(results)} 条记录")
        
        # 显示最新的5条
        print(f"\n最新的5条记录:")
        for i, fact in enumerate(results[:5], 1):
            print(f"\n{i}. Fact ID: {fact['fact_id']}")
            print(f"   时间: {fact['created_at']}")
            print(f"   内容: {fact['content'][:80]}...")
            if fact['entities']:
                print(f"   实体: {', '.join(fact['entities'][:5])}")


def example_4_time_distribution():
    """示例 4: 时间分布统计"""
    print("\n" + "="*80)
    print("示例 4: 时间分布统计")
    print("="*80)
    
    with Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="Aa@123456"
    ) as store:
        # 获取图统计信息
        stats = store.get_graph_statistics()
        
        print(f"\n图统计:")
        print(f"  - 总 Facts 数: {stats['total_facts']}")
        print(f"  - 总关系数: {stats['total_relations']}")
        print(f"  - 最早的 Fact: {stats['earliest_fact']}")
        print(f"  - 最晚的 Fact: {stats['latest_fact']}")
        
        # 获取按天的时间分布
        distribution = store.get_time_distribution(granularity="day")
        
        print(f"\n最近30天的分布 (前10天):")
        for item in distribution[:10]:
            print(f"  {item['time_bucket']}: {item['count']} 条")
        
        # 获取按月的时间分布
        distribution_monthly = store.get_time_distribution(granularity="month")
        
        print(f"\n按月分布:")
        for item in distribution_monthly[:6]:
            print(f"  {item['time_bucket']}: {item['count']} 条")


def example_5_vector_only():
    """示例 5: 仅使用向量检索（语义相似度）"""
    print("\n" + "="*80)
    print("示例 5: 仅使用向量检索（高语义相关性）")
    print("="*80)
    
    config = GauzRagConfig.from_env(Path(".env"))
    pipeline = GauzRagPipeline(config, project_id="demo")
    
    # 只使用向量检索，不使用图检索
    results = pipeline.search_with_time_dimension(
        query="深度学习和神经网络的应用",
        start_time="2024-01-01T00:00:00",
        end_time="2024-12-31T23:59:59",
        top_k=10,
        use_graph=False,  # 不使用图检索
        use_vector=True   # 只使用向量检索
    )
    
    print(f"\n找到 {len(results['vector_results'])} 条语义相关的记录")
    
    # 按相似度排序显示
    sorted_results = sorted(
        results['vector_results'],
        key=lambda x: x['score'],
        reverse=True
    )
    
    print(f"\n相似度最高的5条:")
    for i, fact in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. 相似度: {fact['score']:.3f}")
        print(f"   内容: {fact['content'][:100]}...")
        print(f"   时间: {fact.get('created_at', 'N/A')}")


def example_6_recent_week_timeline():
    """示例 6: 最近一周的时间线回顾"""
    print("\n" + "="*80)
    print("示例 6: 最近一周的时间线回顾")
    print("="*80)
    
    config = GauzRagConfig.from_env(Path(".env"))
    pipeline = GauzRagPipeline(config, project_id="demo")
    
    # 按天查询最近7天
    today = datetime.now().date()
    
    print(f"\n每日记录统计:")
    for i in range(7):
        day = today - timedelta(days=i)
        start = datetime.combine(day, datetime.min.time())
        end = datetime.combine(day, datetime.max.time())
        
        results = pipeline.search_with_time_dimension(
            query="",  # 空查询，返回所有记录
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            top_k=100,
            use_vector=False  # 只用图检索，更快
        )
        
        count = len(results['merged_results'])
        print(f"  {day}: {count} 条记录")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("GauzRag 时间维度检索示例")
    print("="*80)
    
    examples = [
        ("基本时间范围检索", example_1_basic_time_search),
        ("实体过滤检索", example_2_entity_filter),
        ("Neo4j 图检索", example_3_neo4j_only),
        ("时间分布统计", example_4_time_distribution),
        ("向量语义检索", example_5_vector_only),
        ("时间线回顾", example_6_recent_week_timeline),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n输入示例编号（1-6），或按 Enter 运行所有示例:")
    choice = input("> ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(examples):
        # 运行单个示例
        _, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\n❌ 示例执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 运行所有示例
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n❌ 示例 '{name}' 执行失败: {e}")
                import traceback
                traceback.print_exc()
            print("\n" + "-"*80)
    
    print("\n✅ 示例演示完成！")


if __name__ == "__main__":
    main()

