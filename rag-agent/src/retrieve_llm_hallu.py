import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import ipywidgets as widgets
from IPython.display import display
from zhipuai import ZhipuAI

# 设置 ZhipuAI API 密钥
ZHIPU_API_KEY = ''  # 替换

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 加载文档索引和模型
STORAGE_DIR = "./storage"
INDEX_PATH = f"{STORAGE_DIR}/index.faiss"
META_PATH  = f"{STORAGE_DIR}/chunks.json"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TOP_K = 5
SCORE_THRESHOLD = 0.35

def rewrite_query(query):
    # 改写问题，以增加检索多样性
    queries = [
        query,  # 原始问题
        f"请提供与 {query} 相关的信息",  # 同义改写
        f"关于 {query}，请列出相关内容"   # 关键词改写
    ]
    return queries

def multi_query_retrieval(query, index, meta, model, top_k=5):
    rewritten_queries = rewrite_query(query)
    all_results = []
    
    for q in rewritten_queries:
        qvec = embed_query(model, q)
        results = retrieve(index, meta, qvec, top_k)
        all_results.extend(results)
    
    # 合并所有检索到的结果，并去重
    return sorted(all_results, key=lambda x: x["score"], reverse=True)


def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return model

def embed_query(model, query: str) -> np.ndarray:
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    vec = np.asarray(vec, dtype="float32")  # shape: (1, dim)
    return vec

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
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # 从返回的 response 中获取答案
    answer = response.choices[0].message.content
    return answer

# 创建交互式控件
text_input = widgets.Text(placeholder='请输入问题...')
button = widgets.Button(description="检索答案")
output = widgets.Output()

# 加载索引和模型
index, meta = load_index_and_meta()
model = build_model()

def handle_no_relevant_docs(results):
    top1 = results[0]["score"] if results else 0
    if top1 < SCORE_THRESHOLD:
        print(f"⚠️ 相似度过低（top1={top1:.3f} < {SCORE_THRESHOLD}），文档中可能没有相关内容。")
        return True
    return False

def check_consistency(answers):
    # 判断多个答案之间的一致性
    consistent_answer = max(set(answers), key=answers.count)  # 选择最多出现的答案
    return consistent_answer

# 在生成答案后，应用一致性检查
def on_button_click(b):
    query = text_input.value.strip()
    if not query:
        with output:
            print("请输入问题！")
        return

    results = multi_query_retrieval(query, index, meta, model, TOP_K)

    # 如果检索结果不相关，则拒绝生成答案
    if handle_no_relevant_docs(results):
        return

    # 创建 Prompt，拼接用户问题和检索到的文档
    prompt = f"根据以下文档回答问题。如果文档中没有相关内容，请回答“我不知道”。\n"
    for r in results:
        prompt += f"{r['content']} (相似度: {r['score']:.3f})\n"
    
    prompt += f"\n问题：{query}\n回答："

    # 调用 ZhipuAI 生成多个答案
    answers = [generate_answer_zhipu(prompt) for _ in range(3)]  # 生成多个答案
    consistent_answer = check_consistency(answers)
    print(f"生成的一致性答案：\n{consistent_answer}")


# 按钮点击事件
button.on_click(on_button_click)

# 显示控件
display(text_input, button, output)
