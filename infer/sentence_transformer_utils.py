#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence Transformer 工具模块
用于下载模型、计算语义相似度等功能

注意：使用Qwen3-Embedding-4B模型需要transformers>=4.51.0
如果遇到KeyError: 'qwen3'错误，请升级transformers库：
pip install transformers>=4.51.0
"""

import os
import numpy as np
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerSimilarity:
    """Sentence Transformer 语义相似度计算器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", cache_dir: str = "./models"):
        """
        初始化Sentence Transformer模型
        
        Args:
            model_name (str): 模型名称，默认为 "Qwen/Qwen3-Embedding-4B"
            cache_dir (str): 模型缓存目录
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型缓存目录: {cache_dir}")
    
    def download_model(self, force_download: bool = False):
        """
        从Hugging Face下载Sentence Transformer模型
        
        Args:
            force_download (bool): 是否强制重新下载模型
        """
        try:
            logger.info(f"正在下载模型: {self.model_name}")
            
            # 下载并加载模型
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            logger.info(f"✅ 模型 {self.model_name} 下载并加载成功")
            
            # 显示模型信息
            logger.info(f"模型维度: {self.model.get_sentence_embedding_dimension()}")
            logger.info(f"模型最大序列长度: {self.model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"❌ 模型下载失败: {e}")
            raise
    
    def encode_texts(self, texts: Union[str, List[str]], normalize: bool = True, is_query: bool = False) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts (Union[str, List[str]]): 输入文本或文本列表
            normalize (bool): 是否对向量进行L2归一化
            is_query (bool): 是否为查询文本（用于qwen模型的特殊处理）
        
        Returns:
            np.ndarray: 编码后的向量
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 download_model() 方法")
        
        # 确保输入是列表格式
        if isinstance(texts, str):
            texts = [texts]
        
        # logger.info(f"正在embedding {len(texts)} 个文本...")
        
        # 编码文本
        # 对于qwen模型，如果是查询文本需要使用特殊的prompt_name
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": normalize,
            "show_progress_bar": False  # 关闭进度条，减少输出
        }
        
        # 如果是qwen模型且is_query为True，添加prompt_name参数
        if "qwen" in self.model_name.lower() and is_query:
            encode_kwargs["prompt_name"] = "query"
        
        embeddings = self.model.encode(texts, **encode_kwargs)
        
        # logger.info(f"✅ 编码完成，向量维度: {embeddings.shape}")
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str, normalize: bool = True) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            text1 (str): 第一个文本
            text2 (str): 第二个文本
            normalize (bool): 是否对向量进行归一化
        
        Returns:
            float: 相似度分数 (0-1之间，1表示完全相同)
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 download_model() 方法")
        
        # 编码两个文本（将第一个文本作为查询）
        query_embedding = self.encode_texts([text1], normalize=normalize, is_query=True)
        doc_embedding = self.encode_texts([text2], normalize=normalize, is_query=False)
        embeddings = np.vstack([query_embedding, doc_embedding])
        
        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1])
        
        # 如果进行了归一化，余弦相似度就是点积
        # 如果没有归一化，需要除以向量的模长
        if not normalize:
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            similarity = similarity / (norm1 * norm2)
        
        return float(similarity)
    
    def compute_batch_similarity(self, 
                                query_text: str, 
                                candidate_texts: List[str], 
                                normalize: bool = True) -> List[Tuple[str, float]]:
        """
        计算一个查询文本与多个候选文本的相似度
        
        Args:
            query_text (str): 查询文本
            candidate_texts (List[str]): 候选文本列表
            normalize (bool): 是否对向量进行归一化
        
        Returns:
            List[Tuple[str, float]]: (文本, 相似度分数) 的列表，按相似度降序排列
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 download_model() 方法")
        
        # 编码查询文本和所有候选文本
        query_embedding = self.encode_texts([query_text], normalize=normalize, is_query=True)[0]
        candidate_embeddings = self.encode_texts(candidate_texts, normalize=normalize, is_query=False)
        
        # 计算相似度
        similarities = []
        for i, candidate_text in enumerate(candidate_texts):
            if normalize:
                # 归一化后的向量，直接计算点积
                similarity = np.dot(query_embedding, candidate_embeddings[i])
            else:
                # 未归一化的向量，计算余弦相似度
                similarity = np.dot(query_embedding, candidate_embeddings[i])
                norm1 = np.linalg.norm(query_embedding)
                norm2 = np.linalg.norm(candidate_embeddings[i])
                similarity = similarity / (norm1 * norm2)
            
            similarities.append((candidate_text, float(similarity)))
        
        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def find_most_similar(self, 
                         query_text: str, 
                         candidate_texts: List[str], 
                         top_k: int = 5,
                         normalize: bool = True) -> List[Tuple[str, float]]:
        """
        找到与查询文本最相似的top-k个候选文本
        
        Args:
            query_text (str): 查询文本
            candidate_texts (List[str]): 候选文本列表
            top_k (int): 返回前k个最相似的结果
            normalize (bool): 是否对向量进行归一化
        
        Returns:
            List[Tuple[str, float]]: 前k个最相似的文本及其相似度分数
        """
        similarities = self.compute_batch_similarity(query_text, candidate_texts, normalize)
        return similarities[:top_k]
    
    def compute_pairwise_similarity(self, 
                                   texts: List[str], 
                                   normalize: bool = True) -> np.ndarray:
        """
        计算文本列表中所有文本对的相似度矩阵
        
        Args:
            texts (List[str]): 文本列表
            normalize (bool): 是否对向量进行归一化
        
        Returns:
            np.ndarray: 相似度矩阵 (n x n)
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 download_model() 方法")
        
        # 编码所有文本
        embeddings = self.encode_texts(texts, normalize=normalize)
        
        # 计算相似度矩阵
        if normalize:
            # 归一化后的向量，相似度矩阵就是点积矩阵
            similarity_matrix = np.dot(embeddings, embeddings.T)
        else:
            # 未归一化的向量，计算余弦相似度矩阵
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarity_matrix = np.dot(embeddings, embeddings.T) / (norms * norms.T)
        
        return similarity_matrix
    
    def compute_pairs_similarity(self, 
                                text_pairs: List[Tuple[str, str]], 
                                normalize: bool = True) -> List[float]:
        """
        批量计算多个文本对的相似度
        
        Args:
            text_pairs (List[Tuple[str, str]]): 文本对列表
            normalize (bool): 是否对向量进行归一化
        
        Returns:
            List[float]: 每个文本对的相似度分数
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 download_model() 方法")
        
        if not text_pairs:
            return []
        
        # 分离所有文本
        texts1 = [pair[0] for pair in text_pairs]
        texts2 = [pair[1] for pair in text_pairs]

        # 对于qwen模型，假设第一个文本是查询，第二个是文档
        embeddings1 = self.encode_texts(texts1, normalize=normalize, is_query=True)
        embeddings2 = self.encode_texts(texts2, normalize=normalize, is_query=False)

        # 计算对应位置的相似度
        similarities = []
        for i in range(len(text_pairs)):
            if normalize:
                # 归一化后的向量，相似度就是点积
                similarity = np.dot(embeddings1[i], embeddings2[i])
            else:
                # 未归一化的向量，计算余弦相似度
                norm1 = np.linalg.norm(embeddings1[i])
                norm2 = np.linalg.norm(embeddings2[i])
                similarity = np.dot(embeddings1[i], embeddings2[i]) / (norm1 * norm2)
            
            similarities.append(float(similarity))
        
        return similarities


def compute_semantic_similarity(text1: str, 
                               text2: str, 
                               model_name: str = "Qwen/Qwen3-Embedding-4B",
                               normalize: bool = True) -> float:
    """
    计算两个文本的语义相似度（便捷函数）
    
    Args:
        text1 (str): 第一个文本
        text2 (str): 第二个文本
        model_name (str): 模型名称
        normalize (bool): 是否归一化
    
    Returns:
        float: 相似度分数
    """
    similarity_calculator = SentenceTransformerSimilarity(model_name)
    similarity_calculator.download_model()
    return similarity_calculator.compute_similarity(text1, text2, normalize)


def main():
    """主函数，演示Sentence Transformer的使用"""
    print("=" * 60)
    print("Sentence Transformer 语义相似度计算演示")
    print("=" * 60)
    
    # 创建相似度计算器
    similarity_calculator = SentenceTransformerSimilarity()
    
    # 下载模型
    print("\n1. 下载模型...")
    similarity_calculator.download_model()
    
    # 测试文本
    test_texts = [
        "今天天气很好",
        "今天天气不错",
        "今天是个好天气",
        "我喜欢吃苹果",
        "苹果很好吃",
        "机器学习很有趣",
        "人工智能发展很快"
    ]
    
    print("\n2. 计算两两相似度...")
    query_text = "今天天气很好"
    
    for text in test_texts[1:]:
        similarity = similarity_calculator.compute_similarity(query_text, text)
        print(f"'{query_text}' vs '{text}': {similarity:.4f}")
    
    print("\n3. 批量相似度计算...")
    query = "今天天气很好"
    candidates = test_texts[1:]
    
    similarities = similarity_calculator.compute_batch_similarity(query, candidates)
    print(f"查询: '{query}'")
    print("最相似的文本:")
    for text, score in similarities:
        print(f"  {score:.4f}: {text}")
    
    print("\n4. 找到最相似的top-3...")
    top_similar = similarity_calculator.find_most_similar(query, candidates, top_k=3)
    for i, (text, score) in enumerate(top_similar, 1):
        print(f"  {i}. {score:.4f}: {text}")
    
    print("\n5. 计算相似度矩阵...")
    similarity_matrix = similarity_calculator.compute_pairwise_similarity(test_texts[:4])
    print("相似度矩阵:")
    print("文本索引:", list(range(len(test_texts[:4]))))
    for i, row in enumerate(similarity_matrix):
        print(f"文本{i}: {[f'{x:.3f}' for x in row]}")
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    main()
