"""Prompt-writer endpoint policy from MiniMaxRefPack (MIT)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    chat_url: str
    accepts: frozenset
    chat_timeout: int
    classify_timeout: int
    sends_reasoning: bool
    requires_key: bool
    is_openrouter: bool

    @property
    def degrades(self):
        return not {"text", "image", "audio", "video"} <= self.accepts

    def describe(self):
        return "openrouter" if self.is_openrouter else f"local ({self.chat_url})"


def resolve(provider, api_base=""):
    if provider == "local":
        base = api_base.strip().rstrip("/")
        if not base:
            raise ValueError("Set the local API base in AI settings.")
        return Endpoint(f"{base}/chat/completions", frozenset({"text", "image"}),
                        900, 120, False, False, False)
    if provider != "openrouter":
        raise ValueError("Choose local or openrouter in AI settings.")
    return Endpoint("https://openrouter.ai/api/v1/chat/completions",
                    frozenset({"text", "image", "audio", "video"}), 120, 20, True, True, True)
