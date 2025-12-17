import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import ipywidgets as widgets
from IPython.display import display

STORAGE_DIR = "./storage"
INDEX_PATH = f"{STORAGE_DIR}/index.faiss"
META_PATH  = f"{STORAGE_DIR}/chunks.json"

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

TOP_K = 5
SCORE_THRESHOLD = 0.35  # 经验值：后续可调


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
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    vec = np.asarray(vec, dtype="float32")  # shape: (1, dim)
    return vec


def retrieve(index, meta, qvec: np.ndarray, top_k: int):
    scores, idxs = index.search(qvec, top_k)  # scores shape (1, k)
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


# 创建交互式控件
text_input = widgets.Text(placeholder='请输入问题...')
button = widgets.Button(description="检索答案")
output = widgets.Output()

# 加载索引和模型
index, meta = load_index_and_meta()
model = build_model()


def on_button_click(b):
    query = text_input.value.strip()
    if not query:
        with output:
            print("请输入问题！")
        return

    qvec = embed_query(model, query)
    results = retrieve(index, meta, qvec, TOP_K)

    with output:
        # 清空输出区域
        output.clear_output()

        if not results:
            print("没有检索到结果。")
            return

        top1 = results[0]["score"]
        if top1 < SCORE_THRESHOLD:
            print(f"\n⚠️ 相似度过低（top1={top1:.3f} < {SCORE_THRESHOLD}），文档中可能没有相关内容。")
            print("建议：换个问法/加关键词/确认文档是否包含该主题。")
            return

        print(f"\nTop {TOP_K} 检索结果（分数越大越相关）：")
        for r in results:
            print("-" * 60)
            print(f"score={r['score']:.3f} | source={r['source']} | chunk_id={r['chunk_id']}")
            print(r["content"][:300].replace("\n", " "))
        print("-" * 60)


# 按钮点击事件
button.on_click(on_button_click)

# 显示控件
display(text_input, button, output)
