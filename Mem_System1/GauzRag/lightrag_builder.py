"""
GauzRag Builder（兼容层）
引用统一的 GauzRagGraphBuilder
"""
from .lightrag_graph_builder import GauzRagGraphBuilder

# 别名，保持向后兼容
LightRAGBuilder = GauzRagGraphBuilder
GauzRagBuilder = GauzRagGraphBuilder

__all__ = ['LightRAGBuilder', 'GauzRagBuilder', 'GauzRagGraphBuilder']
