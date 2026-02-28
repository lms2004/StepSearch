# pip install -U sentence-transformers
import os
import re
import argparse
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from retrieval_server import get_retriever, Config as RetrieverConfig
from rerank_server import SentenceTransformerCrossEncoder
from cache import SearchCache
from util import normalize_answer

app = FastAPI()

# 全局锁：FAISS GPU 与 retriever/reranker 的 GPU 操作非线程安全，串行化请求
_retriever_lock = threading.Lock()
class SearchRequest(BaseModel):
    # Backward-compatible request schema:
    # - query: single query string
    # - queries: batch queries
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    topk_retrieval: Optional[int] = 10
    topk: Optional[int] = 5
    return_scores: bool = False


class QueryRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    return_scores: bool = False


class EmbedRequest(BaseModel):
    """Request for encoding texts with the retriever's encoder (e.g. E5)."""
    texts: List[str]
    is_passage: bool = True  # True = passage encoding (e.g. "passage: ..."); False = query encoding


def _resolve_queries(request: SearchRequest) -> List[str]:
    """Accept both `query` and `queries` request formats."""
    if request.queries is not None and len(request.queries) > 0:
        return request.queries
    if request.query is not None and str(request.query).strip():
        return [request.query]
    return []


def _to_legacy_doc(doc: Dict) -> Dict:
    """
    Convert retriever doc schema to Search-R1 legacy schema:
    {"id": "...", "contents": "\"title\"\\ncontent"}
    """
    doc_id = doc.get("id", "")
    if "contents" in doc and doc["contents"] is not None:
        contents = str(doc["contents"])
    else:
        title = str(doc.get("title", "")).strip()
        content = str(doc.get("content", "")).strip()
        if title:
            contents = f"\"{title}\"\n{content}"
        else:
            contents = content
    return {"id": str(doc_id), "contents": contents}


def _pack_docs_with_scores(doc_scores: List[Tuple[Dict, float]], topk: int) -> List[Dict]:
    """Convert [(doc, score), ...] to legacy doc list with score."""
    packed = []
    for doc, score in doc_scores[:topk]:
        packed.append({"document": _to_legacy_doc(doc), "score": float(score)})
    return packed

# ----------- Reranker Config Schema -----------
@dataclass
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

# ----------- 新增接口 -----------
@app.post("/retrieve/rerank")
def retrieve_with_rerank(request: SearchRequest):
    global retriever, reranker, dataset_name, retriever_topk, cache, use_cache
    query_list = _resolve_queries(request)
    if not query_list:
        return {"result": []}
    normalized_queries = [normalize_answer(query) for query in query_list]
    # 存储查询结果
    results = []
    # 记录未命中缓存的查询及其索引
    uncached_queries = []
    uncached_indices = []
    
    # 首先检查缓存（可选）
    if use_cache:
        for i, query in enumerate(normalized_queries):
            cached_result = cache.get(query)
            if cached_result is not None:
                # 缓存命中
                results.append(cached_result)
            else:
                # 缓存未命中，添加到待检索列表
                results.append(None)  # 占位
                uncached_queries.append(query_list[i])
                uncached_indices.append(i)
    else:
        # 不使用缓存时，全部进入检索流程
        results = [None] * len(query_list)
        uncached_queries = list(query_list)
        uncached_indices = list(range(len(query_list)))
    
    # 如果有未缓存的查询，执行检索
    if uncached_queries:
        with _retriever_lock:
            # Step 1: 检索文档（FAISS GPU 非线程安全）
            retrieved_docs, scores = retriever.batch_search(
                query_list=uncached_queries,
                num=request.topk_retrieval if request.topk_retrieval else retriever_topk,
                return_score=True
            )
            # Step 2: 重排序（同样使用 GPU，与检索共用一把锁）
            for i, idx in enumerate(uncached_indices):
                single_query = [uncached_queries[i]]
                single_docs = [retrieved_docs[i]]
                reranked = reranker.rerank(single_query, single_docs)
                doc_scores = reranked.get(0, [])
                if request.return_scores:
                    combined = _pack_docs_with_scores(doc_scores, request.topk if request.topk else 5)
                else:
                    combined = [_to_legacy_doc(doc) for doc, _score in doc_scores[: (request.topk if request.topk else 5)]]
                results[idx] = combined
                if use_cache:
                    cache.set(normalized_queries[idx], combined)
    
    return {"result": results}

@app.post("/retrieve")
def retrieve_without_rerank(request: QueryRequest):
    """
    Compatible with examples/search/retriever/retrieval_server.py:
    single query in, wrapped result list out.
    """
    global retriever, retriever_topk

    if not request.topk:
        request.topk = retriever_topk

    with _retriever_lock:
        if request.return_scores:
            results, scores = retriever.search(query=request.query, num=request.topk, return_score=True)
        else:
            results = retriever.search(query=request.query, num=request.topk, return_score=False)
            scores = None

    resp = []
    if request.return_scores and scores is not None:
        combined = []
        for doc, score in zip(results, scores):
            combined.append({"document": _to_legacy_doc(doc), "score": float(score)})
        resp.append(combined)
    else:
        resp.append([_to_legacy_doc(doc) for doc in results])
    return {"result": resp}


@app.post("/embed")
def embed_texts(request: EmbedRequest):
    """
    Encode texts using the retriever's encoder (e.g. intfloat/e5-base-v2).
    Use for information-gain reward: encode gold docs and retrieved docs as passages (is_passage=True).
    Returns embeddings as list of lists for JSON serialization.
    """
    global retriever
    if not request.texts:
        return {"embeddings": []}
    if not hasattr(retriever, "encoder"):
        return {"error": "embed not available: retriever has no encoder", "embeddings": []}
    texts = [t.strip() for t in request.texts if t is not None]
    if not texts:
        return {"embeddings": []}
    with _retriever_lock:
        arr = retriever.encoder.encode(texts, is_query=not request.is_passage)
    embeddings = arr.tolist()
    return {"embeddings": embeddings}


def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")

    # 基础参数
    parser.add_argument("--index_path", type=str, help="索引文件路径")
    parser.add_argument("--corpus_path", type=str, help="语料库文件路径")
    parser.add_argument("--data_root", type=str, default="/mnt/GeneralModel/zhengxuhui/data/stepsearch", help="数据集根目录")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称，用于缓存和初始化")
    parser.add_argument('--faiss_gpu', action='store_true', help='使用GPU进行计算') 
    parser.add_argument('--port', type=int, default=8000, help='端口地址')
    parser.add_argument('--use_cache', action='store_true', help='是否启用缓存（默认关闭）')

    
    # 检索器配置
    parser.add_argument("--retriever_model", type=str, default="/mnt/GeneralModel/wangziliang1/envs/search_r1/search-model/intfloat/e5-base-v2", help="检索器模型路径")
    parser.add_argument("--topk", type=int, default=10, help="检索的默认topk值")
    parser.add_argument("--retriever_name", type=str, default="e5", help="检索器名称")
    
    # 重排序配置
    parser.add_argument("--reranker_model", type=str, default="/mnt/GeneralModel/wangziliang1/envs/search_r1/search-model/cross-encoder/ms-marco-MiniLM-L12-v2", help="重排序模型路径")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="重排序推理的批处理大小")
    parser.add_argument("--reranking_topk", type=int, default=3, help="每个查询重排序的段落数量")

    args = parser.parse_args()
    
    print("[进度] 参数解析完成 (dataset=%s, port=%s, use_cache=%s)." % (args.dataset_name, args.port, args.use_cache))
    
    # 根据数据集名称设置默认值
    if args.dataset_name == "wiki18_e5" or args.dataset_name == "wiki18_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/e5_Flat.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/wiki-18.jsonl"
            
    # elif args.dataset_name == "wiki18_e5_zill" or args.dataset_name == "wiki18_e5_rerank3":
    #     if not args.index_path:
    #         args.index_path = "/mnt/GeneralModel/wangziliang1/data/musi.index"
    #     if not args.corpus_path:
    #         args.corpus_path = "/mnt/GeneralModel/wangziliang1/data/musi.jsonl"
            
    elif args.dataset_name == "musi_e5" or args.dataset_name == "musi_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/musi.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/musi.jsonl"
            
    elif args.dataset_name == "nq_hotpot_e5" or args.dataset_name == "nq_hotpot_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/nq_hotpot.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/nq_hotpot.jsonl"
                        
    elif args.dataset_name == "musi_nq_hotpot_e5" or args.dataset_name == "musi_nq_hotpot_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/musi_nq_hotpot.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/musi_nq_hotpot.jsonl" 

    else:
        print(f"未知的数据集名称: {args.dataset_name}")
        exit(1)
    
    # 1) 初始化检索器配置
    config = RetrieverConfig(
        retrieval_method = args.retriever_name,
        index_path = args.index_path,
        corpus_path = args.corpus_path,
        retrieval_topk = args.topk,
        faiss_gpu = args.faiss_gpu,
        retrieval_model_path = args.retriever_model,
        retrieval_pooling_method = "mean",
        retrieval_query_max_length = 256,
        retrieval_use_fp16 = True,
        retrieval_batch_size = 512,
    )
    
    # 2) 初始化重排序器配置
    reranker_config = RerankerArguments(
        rerank_topk = args.reranking_topk,
        rerank_model_name_or_path = args.reranker_model,
        batch_size = args.reranker_batch_size,
    )
    
    # 3) 实例化全局检索器和重排序器
    print("[进度] 开始加载检索器 (FAISS 索引 + 语料 + 编码器模型)...")
    retriever = get_retriever(config)
    print("[进度] 检索器加载完成.")
    print("[进度] 开始加载重排序模型: %s ..." % (reranker_config.rerank_model_name_or_path,))
    reranker = get_reranker(reranker_config)
    print("[进度] 重排序模型加载完成.")
    dataset_name = args.dataset_name
    retriever_topk = args.topk
    use_cache = args.use_cache
    if use_cache:
        print("[进度] 正在初始化缓存 (dataset=%s) ..." % (dataset_name,))
        cache = SearchCache(
            dataset_name=dataset_name,
        )
        print("[进度] 缓存初始化完成.")
    else:
        cache = None
    # 4) 启动服务器
    print("[进度] 正在启动 HTTP 服务 (host=0.0.0.0, port=%s) ..." % (args.port,))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
