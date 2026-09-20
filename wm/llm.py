"""wm.llm -- LLM 后端的极简抽象。

刻意做得极薄：World Model 架构里 LLM 只是若干可插拔的实现细节之一，
不是架构主体。缺了它系统照样能跑（降级到符号 / 启发式实现），换了它也不用改架构。

支持的规格串：
    None / "off"               → 不联网，全部走符号后备
    "https://api.xxx/v1|<key>|<model>"  → OpenAI 兼容接口
"""
from __future__ import annotations

import json as _json
import urllib.error
import urllib.request
from typing import Optional


class LLMBackend:
    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError


class NullBackend(LLMBackend):
    """离线占位：永远不可用，促使调用方走后备路径。"""

    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        raise RuntimeError("LLM backend 未配置")


class OpenAICompatibleBackend(LLMBackend):
    """零依赖的 OpenAI /chat/completions 调用实现。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.calls = 0

    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        self.calls += 1
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=_json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def get_backend(spec: Optional[str]) -> Optional[LLMBackend]:
    """由规格串构造后端。None 表示离线。"""
    if spec is None or spec in ("", "off", "none"):
        return None
    parts = spec.split("|")
    if len(parts) < 2:
        raise ValueError("规格应为 'base_url|api_key[|model]'")
    base, key = parts[0], parts[1]
    model = parts[2] if len(parts) > 2 and parts[2] else "gpt-4o-mini"
    return OpenAICompatibleBackend(base, key, model)
