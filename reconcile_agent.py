from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

from data_ingest import ensure_id, dataframe_to_texts
from embeddings_index import EmbeddingsIndex


@dataclass
class ReconcileConfig:
    embeddings_model: str
    embeddings_cache: Optional[str]
    index_metric: str
    index_persist_dir: Optional[str]
    auto_merge_threshold: float
    suggest_merge_threshold: float
    llm_backend: str
    llama_model_path: Optional[str]
    llama_n_ctx: int
    llama_n_threads: int
    llama_n_gpu_layers: int


def load_config(path: str = "agent_config.yaml") -> ReconcileConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ReconcileConfig(
        embeddings_model=cfg["embeddings"]["model_name"],
        embeddings_cache=cfg["embeddings"].get("cache_dir"),
        index_metric=cfg["index"]["metric"],
        index_persist_dir=cfg["index"].get("persist_dir"),
        auto_merge_threshold=cfg["reconciliation"]["auto_merge_threshold"],
        suggest_merge_threshold=cfg["reconciliation"]["suggest_merge_threshold"],
        llm_backend=cfg["reconciliation"].get("llm_backend", "simple"),
        llama_model_path=cfg["reconciliation"].get("llama_cpp", {}).get("model_path"),
        llama_n_ctx=int(cfg["reconciliation"].get("llama_cpp", {}).get("n_ctx", 4096)),
        llama_n_threads=int(cfg["reconciliation"].get("llama_cpp", {}).get("n_threads", 6)),
        llama_n_gpu_layers=int(cfg["reconciliation"].get("llama_cpp", {}).get("n_gpu_layers", 0)),
    )


class LocalLLM:
    def __init__(self, backend: str, cfg: ReconcileConfig) -> None:
        self.backend = backend
        self.cfg = cfg
        self.client = None
        if backend == "llama_cpp" and cfg.llama_model_path:
            try:
                from llama_cpp import Llama
                self.client = Llama(
                    model_path=cfg.llama_model_path,
                    n_ctx=cfg.llama_n_ctx,
                    n_threads=cfg.llama_n_threads,
                    n_gpu_layers=cfg.llama_n_gpu_layers,
                )
                logger.info("Initialized llama.cpp backend")
            except Exception as e:
                logger.warning(f"Failed to initialize llama.cpp: {e}. Falling back to simple backend.")
                self.backend = "simple"
        elif backend == "gpt4all":
            try:
                from gpt4all import GPT4All
                model_path = None
                if self.cfg.llama_model_path and self.cfg.llama_model_path.lower().endswith((".gguf", ".bin")):
                    model_path = self.cfg.llama_model_path
                self.client = GPT4All(model_name=model_path) if model_path else GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
                logger.info("Initialized GPT4All backend")
            except Exception as e:
                logger.warning(f"Failed to initialize GPT4All: {e}. Falling back to simple backend.")
                self.backend = "simple"

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> Optional[str]:
        try:
            if self.backend == "llama_cpp" and self.client is not None:
                out = self.client.create_completion(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
                return out["choices"][0]["text"].strip()
            if self.backend == "gpt4all" and self.client is not None:
                return self.client.generate(prompt, temp=temperature, max_tokens=max_tokens)
        except Exception as e:
            logger.warning(f"LLM generate failed: {e}")
        return None

    def score_and_explain(self, record_a: Dict[str, Any], record_b: Dict[str, Any], sim: float) -> Tuple[float, str]:
        if self.backend in {"llama_cpp", "gpt4all"} and self.client is not None:
            prompt = (
                "You are a data reconciliation assistant. Given two records, you output a match score between 0 and 1 "
                "and a short explanation.\n\n"
                f"Record A: {record_a}\n"
                f"Record B: {record_b}\n"
                f"Base similarity: {sim:.3f}\n"
                "Respond as JSON with keys score (0-1 float) and reason (string)."
            )
            text = self.generate(prompt, temperature=0.1, max_tokens=128)
            if text:
                try:
                    import json as _json
                    j = _json.loads(text)
                    score = float(j.get("score", sim))
                    reason = str(j.get("reason", "LLM provided no reason"))
                    return score, reason
                except Exception:
                    logger.warning(f"LLM JSON parse failed, falling back to heuristic. Raw: {text[:200]}")
        reason = (
            "Heuristic decision based on embedding cosine similarity. "
            f"Similarity={sim:.3f}. Consider verifying name/address/ids if present."
        )
        return float(sim), reason


class ReconcileAgent:
    def __init__(self, cfg: Optional[ReconcileConfig] = None) -> None:
        self.cfg = cfg or load_config()
        self.index = EmbeddingsIndex(
            model_name=self.cfg.embeddings_model,
            cache_dir=self.cfg.embeddings_cache,
            persist_dir=self.cfg.index_persist_dir,
            metric=self.cfg.index_metric,
        )
        self.llm = LocalLLM(self.cfg.llm_backend, self.cfg)

    def _build_index(self, texts_b: List[str], ids_b: List[int]) -> None:
        if not self.index.load():
            self.index.build(texts_b, ids=ids_b)
            self.index.save()

    def reconcile(self, df_a, df_b, text_fields: Optional[List[str]] = None, top_k: int = 3) -> Dict[str, Any]:
        df_a = ensure_id(df_a)
        df_b = ensure_id(df_b)

        texts_a = dataframe_to_texts(df_a, text_fields)
        texts_b = dataframe_to_texts(df_b, text_fields)

        ids_b = df_b["_id"].astype(int).tolist()
        self._build_index(texts_b, ids_b)

        sims, idxs = self.index.query(texts_a, top_k=top_k)

        results: List[Dict[str, Any]] = []
        auto_merge_rows = []
        for i, (_sim_row, idx_row) in enumerate(zip(sims, idxs)):
            a_id = int(df_a.iloc[i]["_id"]) 
            a_rec = df_a.iloc[i].to_dict()
            # Take top candidate from B
            best_rank = int(idx_row[0]) if idx_row.size > 0 else -1
            if best_rank < 0:
                results.append({
                    "a_id": a_id,
                    "decision": "no-merge",
                    "score": 0.0,
                    "b_id": None,
                    "reason": "No candidate found"
                })
                continue
            sim = float(sims[i][0])
            b_row = df_b.iloc[best_rank]
            b_id = int(b_row["_id"]) if "_id" in b_row else int(best_rank + 1)
            b_rec = b_row.to_dict()

            score, reason = self.llm.score_and_explain(a_rec, b_rec, sim)

            if score >= self.cfg.auto_merge_threshold:
                decision = "auto-merge"
                merged = {**b_rec, **a_rec}  # prefer A over B on conflict for now
                auto_merge_rows.append(merged)
            elif score >= self.cfg.suggest_merge_threshold:
                decision = "suggest-merge"
                merged = {**b_rec, **a_rec}
            else:
                decision = "no-merge"
                merged = None

            results.append({
                "a_id": a_id,
                "b_id": b_id,
                "score": score,
                "similarity": sim,
                "decision": decision,
                "reason": reason,
                "suggested_merged_record": merged,
            })

        merged_df = None
        if auto_merge_rows:
            import pandas as pd
            merged_df = pd.DataFrame(auto_merge_rows)

        summary = {
            "auto_merged_count": sum(1 for r in results if r["decision"] == "auto-merge"),
            "suggest_merge_count": sum(1 for r in results if r["decision"] == "suggest-merge"),
            "no_merge_count": sum(1 for r in results if r["decision"] == "no-merge"),
        }

        return {
            "summary": summary,
            "results": results,
            "auto_merged_preview": merged_df.head(10).to_dict(orient="records") if merged_df is not None else [],
        }
