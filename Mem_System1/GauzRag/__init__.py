"""
GauzRag: 基于 GraphRAG 的知识图谱 + 语义召回系统

完整的知识处理链路：
  原始文档 → Facts 提取 → 知识图谱构建 → Community 映射 → 语义索引 → 智能召回

作者: Gauz
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Gauz"

from .config import GauzRagConfig
from .pipeline import GauzRagPipeline
from .fact_extractor import FactExtractor
# GraphBuilder 延迟导入（避免 scipy 依赖问题）
from .community_mapper import CommunityMapper
from .embedder import DashScopeEmbedder
from .searcher import CommunitySearcher
from .database import DatabaseManager

# API 可选导入
try:
    from .api import GauzRagAPI, create_app
    __all__ = [
        "GauzRagConfig",
        "GauzRagPipeline",
        "FactExtractor",
        # "GraphBuilder",  # 延迟导入，不在顶层暴露
        "CommunityMapper",
        "DashScopeEmbedder",
        "CommunitySearcher",
        "DatabaseManager",
        "GauzRagAPI",
        "create_app",
    ]
except ImportError:
    # FastAPI 未安装
    __all__ = [
        "GauzRagConfig",
        "GauzRagPipeline",
        "FactExtractor",
        # "GraphBuilder",  # 延迟导入，不在顶层暴露
        "CommunityMapper",
        "DashScopeEmbedder",
        "CommunitySearcher",
        "DatabaseManager",
    ]

