import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import ipywidgets as widgets
from IPython.display import display
from zhipuai import ZhipuAI
from src.agent import RAGAgent, AgentConfig


# 设置 ZhipuAI API 密钥
ZHIPU_API_KEY = ''

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 加载文档索引和模型
STORAGE_DIR = "./storage"
INDEX_PATH = f"{STORAGE_DIR}/index.faiss"
META_PATH  = f"{STORAGE_DIR}/chunks.json"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TOP_K = 5
SCORE_THRESHOLD = 0.35

def recall_at_k(results, ground_truth, k):
    relevant_docs = set(ground_truth)  # 相关文档的 ID
    retrieved_docs = [r['chunk_id'] for r in results[:k]]  # 前 k 个检索到的文档的 ID
    intersection = set(retrieved_docs) & relevant_docs
    recall = len(intersection) / len(relevant_docs) if len(relevant_docs) > 0 else 0
    return recall

def mean_reciprocal_rank(results, ground_truth):
    reciprocal_ranks = []
    for r in results:
        if r['chunk_id'] in ground_truth:
            rank = results.index(r) + 1  # 计算排名（从 1 开始）
            reciprocal_ranks.append(1 / rank)
            break  # 找到第一个相关文档后退出
    return sum(reciprocal_ranks) / len(ground_truth) if reciprocal_ranks else 0


# def rewrite_query(query):
#     # 改写问题，以增加检索多样性
#     queries = [
#         query,  # 原始问题
#         f"请提供与 {query} 相关的信息",  # 同义改写
#         f"关于 {query}，请列出相关内容"   # 关键词改写
#     ]
#     return queries

def rewrite_query(query, mode="normal"):
    if mode == "normal":
        return [
            query,
            f"请提供与 {query} 相关的信息",
            f"关于 {query}，请列出相关内容"
        ]
    # expanded：召回更强，用于 Agent 重试
    return [
        query,
        f"{query} 相关 定义 原理 说明",
        f"请从文档中查找：{query} 的流程/步骤/要点",
        f"{query} 的区别 对比 举例 注意事项"
    ]


# def multi_query_retrieval(query, index, meta, model, top_k=5):
#     rewritten_queries = rewrite_query(query)
#     all_results = []
    
#     for q in rewritten_queries:
#         qvec = embed_query(model, q)
#         results = retrieve(index, meta, qvec, top_k)
#         all_results.extend(results)
    
#     # 合并所有检索到的结果，并去重
#     return sorted(all_results, key=lambda x: x["score"], reverse=True)

def retrieve_evidence(query, mode="normal", top_k=5):
    rewritten_queries = rewrite_query(query, mode=mode)
    all_results = []

    for q in rewritten_queries:
        qvec = embed_query(model, q)
        all_results.extend(retrieve(index, meta, qvec, top_k))

    # 去重：同 chunk_id 只保留最高分
    best = {}
    for r in all_results:
        cid = r["chunk_id"]
        if cid not in best or r["score"] > best[cid]["score"]:
            best[cid] = r

    results = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def build_prompt(query, results):
    evidence = []
    for r in results:
        evidence.append(f"[chunk_id:{r['chunk_id']}] {r['content']}")
    evidence_text = "\n\n".join(evidence)

    return f"""你是一个中文问答助手，只能使用【证据】回答。
规则：
1) 若证据不足或未提及，回答：证据不足，我不知道。
2) 回答中每个关键结论后必须标注引用，格式：[chunk_id:xx]
3) 不要编造，不要用常识补全。

【证据】
{evidence_text}

【问题】
{query}

【回答】
"""


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

def generate_answer(query, results):
    if not results:
        return "证据不足，我不知道。"

    prompt = build_prompt(query, results)
    answers = [generate_answer_zhipu(prompt) for _ in range(3)]
    return check_consistency(answers)


# def on_button_click(b):
#     query = text_input.value.strip()
#     if not query:
#         with output:
#             print("请输入问题！")
#         return

#     # 多查询检索
#     results = multi_query_retrieval(query, index, meta, model, TOP_K)

#     # 评估检索结果
#     ground_truth = [1, 3, 5]  # 假设这是查询的相关文档 ID
#     recall_5 = recall_at_k(results, ground_truth, k=5)
#     mrr_score = mean_reciprocal_rank(results, ground_truth)
    
#     print(f"Recall@5: {recall_5:.4f}")
#     print(f"MRR: {mrr_score:.4f}")

#     # 如果检索结果相似度过低，拒绝生成答案
#     if handle_no_relevant_docs(results):
#         return

#     # 创建 Prompt，拼接用户问题和检索到的文档
#     prompt = f"根据以下文档回答问题。如果文档中没有相关内容，请回答“我不知道”。\n"
#     for r in results:
#         prompt += f"{r['content']} (相似度: {r['score']:.3f})\n"
    
#     prompt += f"\n问题：{query}\n回答："

#     # 调用 ZhipuAI 生成多个答案
#     answers = [generate_answer_zhipu(prompt) for _ in range(3)]  # 生成多个答案
#     consistent_answer = check_consistency(answers)
#     print(f"生成的一致性答案：\n{consistent_answer}")

def on_button_click(b):
    query = text_input.value.strip()
    if not query:
        with output:
            print("请输入问题！")
        return

    with output:
        output.clear_output()

        # 1) 先用 Agent 得到答案
        answer, results, debug = agent.answer(query)

        # 2) 保留检索评估
        # ground_truth = [ ... ]
        # print("Recall@5:", recall_at_k(results, ground_truth, 5))
        # print("MRR:", mean_reciprocal_rank(results, ground_truth))

        print("=== Agent Debug ===")
        print(debug)

        print("\n=== Answer ===")
        print(answer)

        print("\n=== Top Evidence ===")
        for r in results[:3]:
            print(f"- score={r['score']:.3f} | source={r['source']} | chunk_id={r['chunk_id']}")

agent = RAGAgent(
    retrieve_fn=retrieve_evidence,
    generate_fn=generate_answer,
    cfg=AgentConfig(top_k=TOP_K, score_threshold=SCORE_THRESHOLD, max_retries=1),
    llm_direct_fn=None  # 先不做 direct 分支，全部走 RAG
)
    
# 按钮点击事件
button.on_click(on_button_click)

# 显示控件
display(text_input, button, output)
