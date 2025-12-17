# src/agent.py
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Tuple
import re

@dataclass
class AgentConfig:
    top_k: int = 5
    score_threshold: float = 0.35
    max_retries: int = 1  # 只做一次重试就够用

class RAGAgent:
    """
    依赖注入（DI）：
    - retrieve_fn(query, mode:str, top_k:int) -> results(list[dict])
    - generate_fn(query, results) -> answer(str)
    """
    def __init__(
        self,
        retrieve_fn: Callable[[str, str, int], List[Dict[str, Any]]],
        generate_fn: Callable[[str, List[Dict[str, Any]]], str],
        cfg: AgentConfig = AgentConfig(),
        llm_direct_fn: Callable[[str], str] | None = None,
    ):
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.cfg = cfg
        self.llm_direct_fn = llm_direct_fn

    def route(self, query: str) -> str:
        # 规则路由
        need_rag = ["文档", "资料", "根据", "手册", "流程", "章节", "定义", "总结"]
        if any(k in query for k in need_rag):
            return "rag"
        if "如何" in query or "怎么" in query or "区别" in query or "是什么" in query:
            return "rag"
        return "direct" if self.llm_direct_fn else "rag"

    def check(self, answer: str, results: List[Dict[str, Any]]) -> Tuple[bool, str]:
        top1 = results[0]["score"] if results else 0.0
        if top1 < self.cfg.score_threshold:
            return False, f"low_evidence(top1={top1:.3f})"

        # 引用约束：至少出现一个 [chunk_id:xx]
        if not re.search(r"\[chunk_id:\s*\d+\]", answer):
            return False, "missing_citation"

        return True, "ok"

    def answer(self, query: str):
        debug = {"route": None, "attempts": []}
        route = self.route(query)
        debug["route"] = route

        if route == "direct":
            ans = self.llm_direct_fn(query)
            return ans, [], debug

        # RAG + 重试（normal -> expanded）
        for attempt in range(self.cfg.max_retries + 1):
            mode = "normal" if attempt == 0 else "expanded"
            results = self.retrieve_fn(query, mode=mode, top_k=self.cfg.top_k)
            ans = self.generate_fn(query, results)

            ok, reason = self.check(ans, results)
            debug["attempts"].append({
                "attempt": attempt,
                "mode": mode,
                "top1": results[0]["score"] if results else 0.0,
                "check": reason,
            })
            if ok:
                return ans, results, debug

        return "证据不足：文档中没有找到可靠依据，我不确定。", [], debug
