from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import yaml
from loguru import logger

from reconcile_agent import ReconcileAgent, load_config
from data_ingest import load_path


class ReconcileRequest(BaseModel):
    records_a: List[Dict[str, Any]]
    records_b: List[Dict[str, Any]]
    text_fields: Optional[List[str]] = None


# Chatbot models
class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


app = FastAPI(title="Reconciliation Agent", version="0.2.0")


@app.get("/")
async def root():
    return {"message": "Reconciliation Agent is running. POST /reconcile, POST /chat, UI at /chatui"}


@app.post("/reconcile")
async def reconcile(req: ReconcileRequest):
    df_a = pd.DataFrame(req.records_a)
    df_b = pd.DataFrame(req.records_b)

    agent = ReconcileAgent()
    out = agent.reconcile(df_a, df_b, text_fields=req.text_fields)
    return out


@app.post("/chat")
async def chat(req: ChatRequest):
    agent = ReconcileAgent()
    llm = agent.llm

    # Load guard config (with safe defaults)
    try:
        with open("agent_config.yaml", "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
            guard = y.get("chat_guard", {}) or {}
    except Exception as e:
        logger.warning(f"chat_guard config not found or unreadable: {e}")
        guard = {}

    allow_off_topic = bool(guard.get("allow_off_topic", False))
    keywords = guard.get("keywords") or [
        "reconcile", "reconciliation", "matching", "dedupe", "deduplication", "merge", "conflict",
        "faiss", "embedding", "embeddings", "sentence-transformers", "cosine", "index", "similarity",
        "csv", "json", "data quality", "normalization", "threshold", "auto-merge", "suggest-merge",
        "human-in-the-loop", "fastapi", "api", "endpoint", "llama", "gpt4all", "llm", "agent",
    ]
    refusal_message = guard.get("refusal_message", (
        "I’m scoped to reconciliation topics only: ingest/normalize CSV or JSON; embeddings; "
        "FAISS cosine similarity and indexing; merge thresholds (auto/suggest/no-merge); conflict resolution; "
        "human-in-the-loop; and this API. Please ask a reconciliation-related question."
    ))
    gen_cfg = (guard.get("generation") or {})
    temperature = float(gen_cfg.get("temperature", 0.2))
    max_tokens = int(gen_cfg.get("max_tokens", 256))

    # Enforced system prompt (ignore client-provided system)
    base_system = (
        "You are a domain-constrained assistant for a local data reconciliation service. "
        "Only answer questions related to: CSV/JSON ingestion, normalization, embeddings, FAISS, cosine similarity, "
        "indexing, thresholds (auto/suggest/no-merge), conflict resolution, human-in-the-loop, and this API. "
        "If the user asks about anything else, politely refuse and request a reconciliation-related question."
    )

    # Extract messages
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    assistant_msgs = [m.content for m in req.messages if m.role == "assistant"]

    # Topic check over ALL user messages
    lower_all = "\n".join(user_msgs).lower()
    on_topic = any(k in lower_all for k in keywords)

    if not on_topic and not allow_off_topic:
        # Structured log for off-topic refusal
        try:
            from datetime import datetime
            ts = datetime.now().isoformat()
        except Exception:
            ts = "unknown"
        snippet = (" ".join(user_msgs)).strip()
        snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
        logger.info({
            "event": "chat_offtopic_refusal",
            "ts": ts,
            "snippet": snippet,
        })
        return {"reply": refusal_message}

    # Build prompt
    transcript = "\n".join([f"User: {u}" for u in user_msgs] + [f"Assistant: {a}" for a in assistant_msgs])
    prompt = f"{base_system}\n{transcript}\nAssistant:"

    # Use LocalLLM.generate() for any supported backend
    text = llm.generate(prompt, temperature=temperature, max_tokens=max_tokens) if hasattr(llm, "generate") else None
    if text:
        return {"reply": text}

    # Simple fallback when no local LLM backend is configured
    last_user = user_msgs[-1] if user_msgs else ""
    canned = (
        "(Simple mode) Local LLM is not enabled. I can only discuss reconciliation: embeddings, FAISS, thresholds, "
        "auto/suggest/no-merge, conflict resolution, and this API."
    )
    reply = f"{canned}\nYou said: '{last_user}'."
    return {"reply": reply}


@app.get("/chatui", response_class=HTMLResponse)
async def chat_ui():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Local Chatbot</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; }
        #log { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: auto; border-radius: 8px; }
        .bubble { margin: 6px 0; padding: 8px 10px; border-radius: 8px; max-width: 80%; }
        .user { background: #e6f0ff; margin-left: auto; }
        .assistant { background: #f2f2f2; margin-right: auto; }
        #inputRow { display: flex; gap: 8px; margin-top: 12px; }
        #msg { flex: 1; padding: 8px; }
        button { padding: 8px 12px; }
        .note { color: #666; font-size: 0.9em; margin-bottom: 8px; }
      </style>
    </head>
    <body>
      <h2>Local Chatbot</h2>
      <div class="note">The chatbot is restricted to reconciliation topics (ingest/normalize data, embeddings, FAISS, thresholds, merge logic, conflict resolution, and this API).</div>
      <div id="log"></div>
      <div id="inputRow">
        <input id="msg" placeholder="Ask about reconciliation (e.g., How does FAISS similarity work?)" />
        <button id="send">Send</button>
      </div>
      <script>
        const log = document.getElementById('log');
        const msg = document.getElementById('msg');
        const send = document.getElementById('send');
        let history = [{role: 'system', content: 'You are a concise local assistant restricted to reconciliation topics only.'}];

        function addBubble(text, cls) {
          const div = document.createElement('div');
          div.className = 'bubble ' + cls;
          div.textContent = text;
          log.appendChild(div);
          log.scrollTop = log.scrollHeight;
        }

        async function sendMsg() {
          const text = msg.value.trim();
          if (!text) return;
          addBubble(text, 'user');
          msg.value = '';
          history.push({role: 'user', content: text});
          try {
            const resp = await fetch('/chat', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({messages: history})
            });
            const data = await resp.json();
            const reply = data.reply || JSON.stringify(data);
            addBubble(reply, 'assistant');
            history.push({role: 'assistant', content: reply});
          } catch (e) {
            addBubble('Error: ' + e, 'assistant');
          }
        }

        send.addEventListener('click', sendMsg);
        msg.addEventListener('keydown', e => { if (e.key === 'Enter') sendMsg(); });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    cfg = load_config()
    host = "127.0.0.1"
    port = 8000
    try:
        with open("agent_config.yaml", "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
            host = y.get("api", {}).get("host", host)
            port = int(y.get("api", {}).get("port", port))
    except Exception as e:
        logger.warning(f"Could not read agent_config.yaml: {e}")
    import uvicorn
    uvicorn.run(app, host=host, port=port)
