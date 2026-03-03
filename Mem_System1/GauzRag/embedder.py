"""
阿里云百炼 Embedding 封装
使用 OpenAI 兼容接口调用 text-embedding-v4
"""
import os
from typing import List, Union, Optional
import numpy as np
from openai import OpenAI


class DashScopeEmbedder:
    """阿里云百炼 Embedding 客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        初始化 Embedder
        
        Args:
            api_key: API Key，默认从环境变量 DASHSCOPE_API_KEY 读取
            base_url: API Base URL，默认从环境变量 DASHSCOPE_BASE_URL 读取
            model: 模型名称，默认从环境变量 DASHSCOPE_EMBEDDING_MODEL 读取
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
        
        if not self.api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"使用阿里云百炼 Embedding: {self.model}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        batch_size: int = 10
    ) -> Union[List[List[float]], np.ndarray]:
        """
        生成文本的 embedding
        
        Args:
            sentences: 单个文本或文本列表
            show_progress_bar: 是否显示进度条
            convert_to_numpy: 是否转换为 numpy 数组
            batch_size: 批处理大小（阿里云限制最大10）
        
        Returns:
            embedding 向量或向量数组
        """
        # 统一处理为列表
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        total = len(sentences)
        
        # 分批处理
        for i in range(0, total, batch_size):
            batch = sentences[i:i + batch_size]
            
            if show_progress_bar:
                print(f"处理 {i+1}-{min(i+batch_size, total)}/{total}...")
            
            # 调用 API
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # 提取 embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        if convert_to_numpy:
            return np.array(all_embeddings)
        return all_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """返回 embedding 维度"""
        # text-embedding-v4 的维度是 1024
        return 1024

