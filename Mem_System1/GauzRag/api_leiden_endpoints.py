"""
Leiden 社区检测 API 端点
提供社区检测和管理功能的 RESTful 接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os

from .leiden_community_detector import LeidenCommunityDetector
from .lightrag_graph_builder import LightRAGGraphBuilder


router = APIRouter(prefix="/community", tags=["community"])


# ========== 请求模型 ==========

class CommunityDetectionRequest(BaseModel):
    """社区检测请求"""
    project_id: str = Field(..., description="项目 ID")
    resolution: float = Field(1.0, ge=0.1, le=5.0, description="分辨率参数（0.1-5.0，越大社区越多）")
    min_community_size: int = Field(3, ge=1, description="最小社区大小")


class RebuildRelationsRequest(BaseModel):
    """重建关系请求"""
    project_id: str = Field(..., description="项目 ID")
    min_shared_entities: int = Field(2, ge=1, description="最小共享实体数")


# ========== 响应模型 ==========

class CommunityDetectionResponse(BaseModel):
    """社区检测响应"""
    project_id: str
    total_entities: int
    total_relations: int
    total_communities: int
    communities: List[Dict[str, Any]]
    leiden_stats: Dict[str, Any]


class CommunityStatisticsResponse(BaseModel):
    """社区统计响应"""
    project_id: str
    total_communities: int
    total_entities: int
    avg_community_size: float
    largest_community_size: int
    smallest_community_size: int
    unclustered_entities: int


class RebuildRelationsResponse(BaseModel):
    """重建关系响应"""
    project_id: str
    communities_processed: int
    relations_created: int


# ========== API 端点 ==========

@router.post("/detect", response_model=CommunityDetectionResponse)
async def detect_communities(request: CommunityDetectionRequest):
    """
    执行 Leiden 社区检测
    
    核心优势：
    - 降低 Fact 关系匹配复杂度：O(n²) → O(k²)
    - 优化图可视化：避免密集实体图
    - 加速检索：基于社区过滤
    
    参数：
    - resolution: 分辨率（0.1-5.0）
      - 0.5: 粗粒度，社区少、每个社区大
      - 1.0: 标准粒度（推荐）
      - 2.0: 细粒度，社区多、每个社区小
    - min_community_size: 最小社区大小（过滤小社区）
    
    返回：
    - 社区列表及统计信息
    """
    try:
        # 初始化 LightRAG Builder
        builder = LightRAGGraphBuilder(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "Aa@123456"),
            project_id=request.project_id,
            llm_api_key=os.getenv("GAUZ_LLM_API_KEY"),
            llm_api_base=os.getenv("GAUZ_LLM_API_BASE"),
            llm_model=os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini")
        )
        
        try:
            # 执行社区检测
            result = builder.run_leiden_community_detection(
                resolution=request.resolution,
                min_community_size=request.min_community_size
            )
            
            return CommunityDetectionResponse(
                project_id=request.project_id,
                **result
            )
            
        finally:
            await builder.close()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"社区检测失败: {str(e)}"
        )


@router.get("/statistics/{project_id}", response_model=CommunityStatisticsResponse)
async def get_community_statistics(project_id: str):
    """
    获取社区统计信息
    
    返回：
    - 社区数量
    - 平均大小
    - 最大/最小社区
    - 未聚类实体数
    """
    try:
        builder = LightRAGGraphBuilder(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "Aa@123456"),
            project_id=project_id,
            llm_api_key=os.getenv("GAUZ_LLM_API_KEY"),
            llm_api_base=os.getenv("GAUZ_LLM_API_BASE"),
            llm_model=os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini")
        )
        
        try:
            stats = builder.get_community_statistics()
            
            return CommunityStatisticsResponse(
                project_id=project_id,
                **stats
            )
            
        finally:
            await builder.close()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.get("/list/{project_id}")
async def list_communities(
    project_id: str,
    min_size: int = 1,
    limit: int = 100
):
    """
    列出所有社区
    
    参数：
    - min_size: 最小社区大小
    - limit: 返回数量限制
    
    返回：
    - 社区列表，包含 community_id, size, entities
    """
    try:
        detector = LeidenCommunityDetector(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "Aa@123456")
        )
        
        try:
            communities = detector.get_all_communities(
                project_id=project_id,
                min_size=min_size
            )
            
            # 限制返回数量
            communities = communities[:limit]
            
            return {
                "project_id": project_id,
                "total_communities": len(communities),
                "communities": communities
            }
            
        finally:
            detector.close()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"列出社区失败: {str(e)}"
        )


@router.get("/entity/{project_id}/{entity_name}")
async def get_entity_community(project_id: str, entity_name: str):
    """
    查询实体所属的社区
    
    参数：
    - entity_name: 实体名称
    
    返回：
    - community_id 或 null
    """
    try:
        detector = LeidenCommunityDetector(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "Aa@123456")
        )
        
        try:
            community_id = detector.get_entity_community(
                entity_name=entity_name,
                project_id=project_id
            )
            
            if community_id is None:
                return {
                    "entity_name": entity_name,
                    "community_id": None,
                    "message": "实体未分配社区"
                }
            
            # 获取社区内的其他实体
            entities = detector.get_community_entities(
                community_id=community_id,
                project_id=project_id
            )
            
            return {
                "entity_name": entity_name,
                "community_id": community_id,
                "community_size": len(entities),
                "community_entities": entities[:20]  # 只返回前 20 个
            }
            
        finally:
            detector.close()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"查询实体社区失败: {str(e)}"
        )


@router.post("/rebuild_relations", response_model=RebuildRelationsResponse)
async def rebuild_fact_relations(request: RebuildRelationsRequest):
    """
    基于社区重新构建 Fact 关系
    
    只在同一社区内的 Facts 之间建立关系，大幅降低时间复杂度。
    
    参数：
    - min_shared_entities: 最小共享实体数（通常为 2）
    
    优化效果：
    - 全局模式：O(n²)
    - 社区模式：O(k²)，k << n
    
    返回：
    - 处理的社区数
    - 创建的关系数
    """
    try:
        builder = LightRAGGraphBuilder(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "Aa@123456"),
            project_id=request.project_id,
            llm_api_key=os.getenv("GAUZ_LLM_API_KEY"),
            llm_api_base=os.getenv("GAUZ_LLM_API_BASE"),
            llm_model=os.getenv("GAUZ_LLM_MODEL", "gpt-4o-mini")
        )
        
        try:
            result = builder.rebuild_fact_relations_by_community(
                min_shared_entities=request.min_shared_entities
            )
            
            return RebuildRelationsResponse(
                project_id=request.project_id,
                **result
            )
            
        finally:
            await builder.close()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"重建关系失败: {str(e)}"
        )


@router.delete("/clear/{project_id}")
async def clear_communities(project_id: str):
    """
    清除社区信息
    
    删除：
    - Community 节点
    - Entity 节点的 community_id 属性
    - BELONGS_TO 关系
    
    注意：不会删除实体和 Facts，只删除社区信息
    """
    try:
        detector = LeidenCommunityDetector(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "Aa@123456")
        )
        
        try:
            with detector.driver.session(database=detector.database) as session:
                # 删除 Community 节点和关系
                session.run("""
                    MATCH (c:Community {project_id: $project_id})
                    DETACH DELETE c
                """, project_id=project_id)
                
                # 清除 Entity 的 community_id
                result = session.run("""
                    MATCH (e:Entity {project_id: $project_id})
                    REMOVE e.community_id
                    RETURN count(e) AS cleared_count
                """, project_id=project_id).single()
                
                cleared_count = result['cleared_count'] if result else 0
            
            return {
                "project_id": project_id,
                "message": "社区信息已清除",
                "entities_cleared": cleared_count
            }
            
        finally:
            detector.close()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清除社区失败: {str(e)}"
        )

