import gradio as gr
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from zhipuai import ZhipuAI
import os

# ZhipuAI API 密钥
ZHIPU_API_KEY = ''
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 加载模型和索引
STORAGE_DIR = "./storage"
INDEX_PATH = f"{STORAGE_DIR}/index.faiss"
META_PATH  = f"{STORAGE_DIR}/chunks.json"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TOP_K = 5
SCORE_THRESHOLD = 0.35

def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def build_model():
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    return model

def embed_query(model, query: str) -> np.ndarray:
    vec = model.encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

def retrieve(index, meta, qvec: np.ndarray, top_k: int):
    scores, idxs = index.search(qvec, top_k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()
    results = []
    for s, i in zip(scores, idxs):
        if i < 0:
            continue
        item = meta[i]
        results.append({
            "score": float(s),
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "content": item["content"]
        })
    return results

# 使用 ZhipuAI 生成答案
def generate_answer_zhipu(prompt):
    response = client.chat.completions.create(
        model="glm-4-flash-250414", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rewrite_query(query):
    return [query, f"请提供与 {query} 相关的信息", f"关于 {query}，请列出相关内容"]

def multi_query_retrieval(query, index, meta, model, top_k=5):
    rewritten_queries = rewrite_query(query)
    all_results = []
    
    for q in rewritten_queries:
        qvec = embed_query(model, q)
        results = retrieve(index, meta, qvec, top_k)
        all_results.extend(results)
    
    return sorted(all_results, key=lambda x: x["score"], reverse=True)

def handle_no_relevant_docs(results):
    top1 = results[0]["score"] if results else 0
    if top1 < SCORE_THRESHOLD:
        return True
    return False

def check_consistency(answers):
    consistent_answer = max(set(answers), key=answers.count)
    return consistent_answer

def answer_question(query):
    # 加载索引和模型
    index, meta = load_index_and_meta()
    model = build_model()

    results = multi_query_retrieval(query, index, meta, model, TOP_K)

    # 如果检索结果不相关，则拒绝生成答案
    if handle_no_relevant_docs(results):
        return "未找到相关文档或相关度过低，请尝试重新提问。"

    # 拼接检索结果，作为生成答案的输入
    prompt = f"根据以下文档回答问题。如果文档中没有相关内容，请回答“我不知道”。\n"
    for r in results:
        prompt += f"{r['content']} (相似度: {r['score']:.3f})\n"
    
    prompt += f"\n问题：{query}\n回答："

    # 生成多个答案并选择一致性答案
    answers = [generate_answer_zhipu(prompt) for _ in range(3)]  # 生成多个答案
    consistent_answer = check_consistency(answers)
    return consistent_answer

# Gradio 界面定义
iface = gr.Interface(fn=answer_question, inputs="text", outputs="text")

# 启动 Gradio Web 应用
# iface.launch()
iface.launch(share=True)

