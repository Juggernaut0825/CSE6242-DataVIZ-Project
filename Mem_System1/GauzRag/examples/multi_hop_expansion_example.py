"""
多跳图谱扩展示例

演示如何使用多跳图谱扩展功能来发现深层关系
"""
import requests
import json
from typing import Dict, Any


def search_with_multi_hop_expansion(
    query: str,
    project_id: str,
    max_hops: int = 2,
    top_k: int = 5,
    use_refine: bool = False,
    api_url: str = "http://localhost:1234"
) -> Dict[str, Any]:
    """
    使用多跳图谱扩展进行搜索
    
    Args:
        query: 查询文本
        project_id: 项目 ID
        max_hops: 最大跳数（1=单跳，2=双跳，3=三跳等）
        top_k: 返回结果数量
        use_refine: 是否使用 LLM 精炼
        api_url: API 服务地址
    
    Returns:
        搜索结果
    """
    endpoint = f"{api_url}/search"
    
    payload = {
        "query": query,
        "project_id": project_id,
        "top_k": top_k,
        "use_refine": use_refine,
        "use_graph_expansion": True,  # 启用图谱扩展
        "max_hops": max_hops  # 设置跳数
    }
    
    print(f"\n{'='*80}")
    print(f"🔍 查询: {query}")
    print(f"📊 配置: max_hops={max_hops}, top_k={top_k}, use_graph_expansion=True")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # 打印结果统计
        print(f"✅ 查询成功！")
        print(f"   - Bundles 数量: {result.get('total_bundles', 0)}")
        
        if result.get('bundles'):
            total_facts = sum(len(b.get('facts', [])) for b in result['bundles'])
            total_convs = sum(len(b.get('conversations', [])) for b in result['bundles'])
            total_topics = sum(len(b.get('topics', [])) for b in result['bundles'])
            
            print(f"   - 总 Facts: {total_facts}")
            print(f"   - 总 Conversations: {total_convs}")
            print(f"   - 总 Topics: {total_topics}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return {}


def compare_hop_results(
    query: str,
    project_id: str,
    max_hops_list: list = [1, 2, 3],
    api_url: str = "http://localhost:1234"
):
    """
    对比不同跳数的结果差异
    
    Args:
        query: 查询文本
        project_id: 项目 ID
        max_hops_list: 要测试的跳数列表
        api_url: API 服务地址
    """
    print(f"\n{'='*80}")
    print(f"📊 对比不同跳数的扩展效果")
    print(f"{'='*80}")
    print(f"查询: {query}")
    print(f"项目: {project_id}\n")
    
    results_comparison = []
    
    for hops in max_hops_list:
        print(f"\n--- 测试 {hops} 跳扩展 ---")
        result = search_with_multi_hop_expansion(
            query=query,
            project_id=project_id,
            max_hops=hops,
            top_k=5,
            api_url=api_url
        )
        
        if result:
            bundles = result.get('bundles', [])
            total_facts = sum(len(b.get('facts', [])) for b in bundles)
            
            results_comparison.append({
                'hops': hops,
                'bundles': len(bundles),
                'facts': total_facts
            })
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print(f"📊 结果对比")
    print(f"{'='*80}")
    print(f"{'跳数':<10} {'Bundles':<15} {'Facts':<15} {'增长率':<15}")
    print(f"{'-'*80}")
    
    for i, comp in enumerate(results_comparison):
        if i == 0:
            growth = "-"
        else:
            prev_facts = results_comparison[i-1]['facts']
            if prev_facts > 0:
                growth_rate = ((comp['facts'] - prev_facts) / prev_facts) * 100
                growth = f"+{growth_rate:.1f}%"
            else:
                growth = "-"
        
        print(f"{comp['hops']:<10} {comp['bundles']:<15} {comp['facts']:<15} {growth:<15}")
    
    print(f"{'='*80}\n")


def example_1_basic_multi_hop():
    """示例 1: 基础多跳查询"""
    print("\n" + "="*80)
    print("示例 1: 基础多跳查询")
    print("="*80)
    
    # 单跳查询
    result_1hop = search_with_multi_hop_expansion(
        query="机器学习的应用",
        project_id="demo",
        max_hops=1,
        top_k=5
    )
    
    # 双跳查询
    result_2hop = search_with_multi_hop_expansion(
        query="机器学习的应用",
        project_id="demo",
        max_hops=2,
        top_k=5
    )
    
    # 三跳查询
    result_3hop = search_with_multi_hop_expansion(
        query="机器学习的应用",
        project_id="demo",
        max_hops=3,
        top_k=5
    )


def example_2_deep_reasoning():
    """示例 2: 深度推理查询（多跳有助于发现间接关系）"""
    print("\n" + "="*80)
    print("示例 2: 深度推理查询")
    print("="*80)
    
    # 这类查询通常需要多跳才能找到完整的推理链
    queries = [
        "GPT-4 对自然语言处理领域的影响",
        "深度学习和图像识别的关系",
        "Transformer 架构为什么重要"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        result = search_with_multi_hop_expansion(
            query=query,
            project_id="demo",
            max_hops=3,  # 使用 3 跳发现深层关系
            top_k=5,
            use_refine=True  # 使用 LLM 精炼结果
        )


def example_3_comparison():
    """示例 3: 对比不同跳数的效果"""
    print("\n" + "="*80)
    print("示例 3: 对比不同跳数的效果")
    print("="*80)
    
    compare_hop_results(
        query="人工智能的发展历程",
        project_id="demo",
        max_hops_list=[1, 2, 3, 4]
    )


def example_4_time_and_hops():
    """示例 4: 结合时间维度和多跳扩展"""
    print("\n" + "="*80)
    print("示例 4: 结合时间维度和多跳扩展")
    print("="*80)
    
    endpoint = "http://localhost:1234/search"
    
    payload = {
        "query": "最近的技术突破",
        "project_id": "demo",
        "top_k": 10,
        "use_graph_expansion": True,
        "max_hops": 2,
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-12-31T23:59:59"
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ 查询成功！")
        print(f"   - Bundles: {result.get('total_bundles', 0)}")
        
        if result.get('bundles'):
            # 显示前3个 Bundle
            for i, bundle in enumerate(result['bundles'][:3], 1):
                print(f"\n📦 Bundle {i}:")
                print(f"   Facts: {len(bundle.get('facts', []))}")
                
                # 显示前3个 Facts
                for j, fact in enumerate(bundle.get('facts', [])[:3], 1):
                    print(f"   {j}. {fact['content'][:80]}...")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("多跳图谱扩展示例")
    print("="*80)
    
    examples = [
        ("基础多跳查询", example_1_basic_multi_hop),
        ("深度推理查询", example_2_deep_reasoning),
        ("对比不同跳数", example_3_comparison),
        ("时间+多跳组合", example_4_time_and_hops),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n输入示例编号（1-4），或按 Enter 运行所有示例:")
    choice = input("> ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(examples):
        _, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\n❌ 示例执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
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

