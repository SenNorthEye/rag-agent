import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DATA_DIR = "./data"
STORAGE_DIR = "./storage"

# 中文 embedding：轻量、CPU可
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 分块参数（Day1 用简单稳定的）
CHUNK_SIZE = 500      # 每个 chunk 大约 500 个字符
CHUNK_OVERLAP = 80


@dataclass
class Chunk:
    content: str
    source: str
    chunk_id: int


def load_documents(data_dir: str) -> List[Dict]:
    docs = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".txt") or fn.endswith(".md"):
            path = os.path.join(data_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"source": fn, "text": text})
    return docs


def normalize_text(text: str) -> str:
    # 简单清洗：去掉多余空白
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    简单可控的中文分块：
    - 先按空行分段
    - 再把段落拼成 chunk，超过 chunk_size 就切
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""

    def flush_buffer(b: str):
        b = b.strip()
        if b:
            chunks.append(b)

    for p in paragraphs:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            flush_buffer(buf)
            buf = p

    flush_buffer(buf)

    # 加 overlap：用字符方式做简单重叠
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        prev_tail = ""
        for c in chunks:
            head = c
            merged = (prev_tail + head).strip()
            overlapped.append(merged)
            prev_tail = head[-overlap:]
        chunks = overlapped

    return chunks


def build_chunks(docs: List[Dict]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in docs:
        text = normalize_text(doc["text"])
        pieces = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(content=piece, source=doc["source"], chunk_id=i))
    return all_chunks


def embed_chunks(model: SentenceTransformer, chunks: List[Chunk]) -> np.ndarray:
    texts = [c.content for c in chunks]
    # normalize_embeddings=True 让相似度更稳定（余弦相似度）
    vecs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    vecs = np.asarray(vecs, dtype="float32")
    return vecs


def save_faiss_index(vectors: np.ndarray, out_dir: str) -> Tuple[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    dim = vectors.shape[1]

    # 因为做了 normalize_embeddings，所以可以用 inner product 近似 cosine
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = os.path.join(out_dir, "index.faiss")
    faiss.write_index(index, index_path)
    return index_path, dim


def save_metadata(chunks: List[Chunk], out_dir: str) -> str:
    meta = [
        {"source": c.source, "chunk_id": c.chunk_id, "content": c.content}
        for c in chunks
    ]
    meta_path = os.path.join(out_dir, "chunks.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


def main():
    docs = load_documents(DATA_DIR)
    if not docs:
        raise RuntimeError(f"在 {DATA_DIR}/ 里没找到 .txt 或 .md 文件，请先放入中文文档。")

    print(f"Loaded docs: {len(docs)}")
    chunks = build_chunks(docs)
    print(f"Total chunks: {len(chunks)}")

    # 打印 2 个 chunk 确认切分效果
    for i in range(min(2, len(chunks))):
        c = chunks[i]
        print("\n--- Sample chunk ---")
        print(f"source={c.source}, chunk_id={c.chunk_id}, len={len(c.content)}")
        print(c.content[:200], "..." if len(c.content) > 200 else "")

    print("\nLoading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    vectors = embed_chunks(model, chunks)
    print(f"\nEmbedding dim: {vectors.shape[1]}")

    index_path, dim = save_faiss_index(vectors, STORAGE_DIR)
    meta_path = save_metadata(chunks, STORAGE_DIR)

    print("\nSaved index to:", index_path)
    print("Saved metadata to:", meta_path)
    print("\n✅文档→分块→向量→FAISS索引 已完成")


if __name__ == "__main__":
    main()
