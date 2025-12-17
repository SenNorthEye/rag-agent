### 基于 Agent 的中文 RAG 问答系统

一、ingest_build_index.py

运行ingest_build_index.py，加载data中的文档信息，切块，计算embedding，存入faiss

##### 1) **Embedding 和向量化**：

- **Embedding** 是将文本转换成 **向量表示**，使得文本的语义可以被模型理解和计算。
- 向量化的文本可以用来进行相似度计算，帮助模型在 **知识库问答系统** 中检索相关信息。

##### 2) **文档分块**：

- 文档太长无法一次性喂给大模型处理，**chunking** 就是把长文档切成小块，每块有一定的 **重叠部分**，从而保证信息的完整性和检索的精度。

##### 3) **FAISS（向量库）**：

- **FAISS** 是一个高效的 **向量检索库**，可以快速搜索相似度高的向量。使用 FAISS 创建了一个索引，使得后续的 **相似度检索** 能够快速返回与用户问题相关的文档块。



二、retrieve.py：基础检索思路

1. 载入保存的 `storage/index.faiss` 和 `storage/chunks.json`
2. 把用户问题做 embedding（GPU）
3. 用 FAISS 找到 topK chunks
4. 输出：**相似度分数 + chunk 内容 + 来源文件名**
5. 加一个 **相似度阈值**：太低就“拒答/提示换问法”



三、retrieve_llm.py：接入大模型，拼接prompt，调用API

```python
    #创建 Prompt，拼接用户问题和检索到的文档
    prompt = f"""
    根据以下文档回答问题。如果文档中没有相关内容，请回答“我不知道”。
    
    文档：
    """
    for r in results:
        prompt += f"{r['content']} (相似度: {r['score']:.3f})\n"
    
    prompt += f"\n问题：{query}\n回答："

    # 调用 ZhipuAI 生成答案
    answer = generate_answer_zhipu(prompt)
    print(f"生成的答案：\n{answer}")
```


四、retrieve_llm_hallu.py：优化RAG系统，减少幻觉问题

**1. 拒答策略（Rejecting Irrelevant Answers）**：

- 当 **检索结果** 的 **相似度低于设定的阈值** 时，模型直接拒绝生成答案。
- 通过设定 **相似度阈值**（如 `SCORE_THRESHOLD = 0.35`），确保只有在找到相关文档时才会生成答案，否则返回“未找到相关内容”。

```python
def handle_no_relevant_docs(results):
    top1 = results[0]["score"] if results else 0
    if top1 < SCORE_THRESHOLD:
        print(f"⚠️ 相似度过低，拒绝生成答案")
        return True
    return False
```

**2. 多查询检索（Multi-query Retrieval）**：

- 对用户问题进行 **多次改写**，从不同角度进行 **查询**，并结合所有查询的检索结果来提升答案的准确性。
- 通过多轮检索，可以增加相关文档的覆盖范围，避免遗漏信息。

```python
def rewrite_query(query):
    return [
        query,  # 原始问题
        f"关于 {query} 的相关信息",  # 改写后的查询
        f"请列出与 {query} 相关的内容"  # 同义改写查询
    ]
```

**3. 自我一致性（Self-Consistency）**：

- 通过生成多个答案并对比其一致性，确保最终答案更加可靠。如果多个答案一致，则选择一致性最强的答案。
- 通过 **多次生成** 和 **对比答案一致性**，提高了生成答案的可信度。

```python
def check_consistency(answers):
    return max(set(answers), key=answers.count)  # 选择最一致的答案
```

优化 **RAG 系统**，确保生成的答案更加可靠和准确，重点解决 **幻觉问题**。

**拒答策略**、**多查询检索** 和 **自我一致性** 是主要的优化方法，它们可以提高 **检索结果** 和 **生成答案的可信度**。



五、retrieve_eva.py：添加评估函数

将 **`Recall@K`** 和 **`MRR`** 函数添加到现有代码中： 

```python
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

```

#### **Recall@K：**

**Recall@K** 衡量的是 **检索到的前 K 个结果** 中，有多少个是 **与用户查询相关的文档**。

​            $\text{Recall@K} = \frac{\text{相关文档的数量}}{\text{总相关文档的数量}}$



#### **MRR (Mean Reciprocal Rank)：**

**MRR** 衡量的是 **检索系统** 在 **相关文档** 出现的 **排名位置** 上的平均性能。

​            $\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$

其中：

- N 是查询的总数。
- $\text{rank}_i$是第 i个查询中相关文档的排名。



六、gradio_web.py：实现web交互页面

```python
import gradio as gr

def answer_question(query):
    # 已经写好的检索和生成答案的逻辑
    # ...
    return consistent_answer

# 创建 Gradio 界面
iface = gr.Interface(fn=answer_question, inputs="text", outputs="text")

# 启动 Gradio Web 应用
iface.launch()
```

1. **实现文档上传功能**：
   - 让用户可以上传自己的文档，然后将文档切分成块，生成向量并存储到索引中。
2. **集成检索和生成功能**：
   - 用户输入问题后，触发 **检索** 和 **生成** 流程，显示 **检索结果** 和 **生成的答案**。
3. **展示界面**：
   - 显示 **检索结果**，并将生成的答案展示给用户。
   - 提供清晰的 **用户输入界面**，确保用户体验友好。



七、agent.py：加入多Agent

对应修改retrieve_evalue.py。（运行这个文件）

**明确做了 4 类 Agent 能力**。

#### 1️⃣ 路由 Agent（Router Agent）

- 根据用户问题内容，判断：
  - 是不是必须基于文档回答（走 RAG）
  - 是否需要检索（而不是直接让大模型自由生成）

------

#### 2️⃣ 重试 Agent（Retry / Re-retrieval Agent）

- 当第一次检索证据不足时：
  - 不直接回答
  - 自动 **换检索策略再来一次**

------

#### 3️⃣ 审校 Agent（Judge / Guard Agent）


对生成结果进行 **自动审查**：

- ✅ 证据是否足够（top1 相似度阈值）
- ✅ 答案是否真的“基于证据”
  - 是否包含 `[chunk_id: xx]` 引用
- ❌ 不满足 → 判失败 → 触发重试 / 拒答

------

#### 4️⃣ 编排 Agent（Orchestration / State Machine）

- 把 RAG 的各个模块**串成一个有状态流程**：

```
ROUTE → RAG → CHECK → (RETRY) → FINAL / REFUSE
```



> **RAG 负责“能力”，Agent 负责“决策与稳定性”。**

- RAG：能不能检索、能不能生成
- Agent：**什么时候检索、失败怎么办、能不能回答**



#### 完整系统流程：

```
用户问题
  ↓
Agent Router（是否需要检索）
  ↓
RAG 检索（多查询 + 去重 + FAISS）
  ↓
证据评估（相似度阈值）
  ↓
LLM 生成（强引用 Prompt + 多次生成一致性）
  ↓
Agent Judge（是否有引用 / 是否证据充分）
  ↓
  ├─ 通过 → 返回答案 + 引用
  ├─ 失败 → 切换检索策略重试
  └─ 多次失败 → 拒答（证据不足）
```

所有代码在云服务器的jupyter中运行。