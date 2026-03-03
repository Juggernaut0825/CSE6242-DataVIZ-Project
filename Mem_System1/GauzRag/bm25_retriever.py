"""
BM25检索器
支持混合检索（向量 + BM25）
"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import jieba
import re
import pickle
from pathlib import Path
import fcntl  # Unix 文件锁
import time
import sys

# 🔥 全局初始化 Jieba（避免每次加载耗时 1 秒）
print("[BM25] 初始化 Jieba 分词器...")
jieba.initialize()
print("[BM25] Jieba 初始化完成")


class BM25Retriever:
    """BM25检索器（支持中英文）"""
    
    def __init__(self, corpus: List[Dict[str, Any]] = None):
        """
        初始化BM25检索器
        
        Args:
            corpus: 文档列表 [{'id': fact_id, 'text': content}, ...]
        """
        self.corpus = corpus or []
        self.bm25 = None
        self.doc_ids = []
        
        if corpus:
            self._build_index(corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词（中英文混合）
        
        策略：
        - 英文：按空格+小写+去停用词
        - 中文：jieba分词
        """
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 检测是否主要是中文
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        is_chinese = chinese_chars / max(total_chars, 1) > 0.3
        
        if is_chinese:
            # 中文分词
            tokens = list(jieba.cut_for_search(text))
            tokens = [t.strip().lower() for t in tokens if len(t.strip()) > 1]
        else:
            # 英文分词
            tokens = text.lower().split()
            # 简单停用词过滤
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of'}
            tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        
        return tokens
    
    def _build_index(self, corpus: List[Dict[str, Any]]):
        """构建BM25索引"""
        self.corpus = corpus
        self.doc_ids = [doc['id'] for doc in corpus]
        
        # 分词
        tokenized_corpus = [self._tokenize(doc['text']) for doc in corpus]
        
        # 构建BM25索引
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def add_documents(self, new_docs: List[Dict[str, Any]]):
        """增量添加文档（重建索引）"""
        self.corpus.extend(new_docs)
        self._build_index(self.corpus)
    
    def save(self, file_path: str):
        """
        持久化BM25索引（带文件锁，防止并发冲突）
        
        Args:
            file_path: 保存路径（.pkl）
        """
        save_path = Path(file_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 只保存corpus数据，加载时重建索引
        data = {
            'corpus': self.corpus,
            'doc_ids': self.doc_ids
        }
        
        # 🔒 使用文件锁防止并发写入
        lock_path = save_path.with_suffix('.lock')
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # 创建锁文件
                lock_file = open(lock_path, 'w')
                
                # Windows 兼容性检查
                if sys.platform == 'win32':
                    # Windows 使用 msvcrt 或跳过锁
                    try:
                        import msvcrt
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    except (ImportError, OSError):
                        # 如果无法导入或锁定失败，继续（降级处理）
                        pass
                else:
                    # Unix/Linux 使用 fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # 获取锁后，执行写入
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                
                print(f"[BM25] 索引已保存: {save_path} ({len(self.corpus)} docs)")
                
                # 释放锁
                if sys.platform != 'win32':
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                lock_path.unlink(missing_ok=True)
                return
                
            except (IOError, BlockingIOError):
                # 锁被占用，等待重试
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)  # 指数退避
                    time.sleep(wait_time)
                else:
                    print(f"[BM25] ❌ 无法获取文件锁，保存失败")
                    raise
            finally:
                try:
                    lock_file.close()
                except:
                    pass
    
    @classmethod
    def load(cls, file_path: str) -> 'BM25Retriever':
        """
        加载持久化的BM25索引
        
        Args:
            file_path: 索引文件路径
        
        Returns:
            BM25Retriever实例
        """
        load_path = Path(file_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"BM25索引文件不存在: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        # 创建实例并重建索引
        retriever = cls(corpus=data['corpus'])
        print(f"[BM25] 索引已加载: {load_path} ({len(retriever.corpus)} docs)")
        
        return retriever
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25搜索
        
        Args:
            query: 查询文本
            top_k: 返回Top K结果
        
        Returns:
            [{'id': fact_id, 'score': bm25_score}, ...]
        """
        if not self.bm25:
            return []
        
        # 查询分词
        query_tokens = self._tokenize(query)
        
        # BM25打分
        scores = self.bm25.get_scores(query_tokens)
        
        # 排序并返回Top K
        doc_scores = list(zip(self.doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = [
            {'id': doc_id, 'score': float(score)}
            for doc_id, score in doc_scores[:top_k]
            if score > 0  # 过滤0分结果
        ]
        
        return results


class HybridRetriever:
    """混合检索器（向量 + BM25）"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        RRF（Reciprocal Rank Fusion）融合算法
        
        RRF Score = Σ (weight / (k + rank))
        
        Args:
            vector_results: 向量检索结果 [{'id': x, 'score': y}, ...]
            bm25_results: BM25检索结果
            k: RRF参数（默认60）
            vector_weight: 向量检索权重
            bm25_weight: BM25权重
        
        Returns:
            融合后的结果
        """
        # 计算RRF分数
        rrf_scores = {}
        
        # 向量检索贡献
        for rank, item in enumerate(vector_results, 1):
            doc_id = item['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight / (k + rank)
        
        # BM25贡献
        for rank, item in enumerate(bm25_results, 1):
            doc_id = item['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight / (k + rank)
        
        # 排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建返回结果
        final_results = [
            {'id': doc_id, 'score': score}
            for doc_id, score in sorted_results
        ]
        
        return final_results
    
    @staticmethod
    def weighted_score_fusion(
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        加权分数融合
        
        Final Score = vector_weight * normalize(vector_score) + bm25_weight * normalize(bm25_score)
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            vector_weight: 向量权重
            bm25_weight: BM25权重
        """
        # 归一化向量分数
        vector_map = {}
        if vector_results:
            max_vec = max(r['score'] for r in vector_results)
            min_vec = min(r['score'] for r in vector_results)
            scale = max_vec - min_vec if max_vec > min_vec else 1.0
            for r in vector_results:
                normalized = (r['score'] - min_vec) / scale if scale > 0 else 0
                vector_map[r['id']] = normalized
        
        # 归一化BM25分数
        bm25_map = {}
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results)
            scale = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            for r in bm25_results:
                normalized = (r['score'] - min_bm25) / scale if scale > 0 else 0
                bm25_map[r['id']] = normalized
        
        # 融合分数
        all_ids = set(vector_map.keys()) | set(bm25_map.keys())
        fused_scores = {}
        for doc_id in all_ids:
            vec_score = vector_map.get(doc_id, 0)
            bm_score = bm25_map.get(doc_id, 0)
            fused_scores[doc_id] = vector_weight * vec_score + bm25_weight * bm_score
        
        # 排序
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'id': doc_id, 'score': score}
            for doc_id, score in sorted_results
        ]

