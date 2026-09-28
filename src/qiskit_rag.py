"""Small local BM25 retriever for Qiskit mutation guidance."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "qiskit_rag"
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


@dataclass(frozen=True)
class RetrievedDocument:
    source: str
    text: str
    score: float


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def retrieve(query: str, top_k: int = 3) -> list[RetrievedDocument]:
    """Rank local Qiskit notes against a per-mutation query using BM25."""
    if top_k < 1:
        return []
    paths = sorted(DOCS_DIR.glob("*.md"))
    documents = [(path.name, path.read_text(encoding="utf-8")) for path in paths]
    if not documents:
        return []

    query_terms = _tokens(query)
    tokenized = [_tokens(text) for _, text in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized)
        for term in set(query_terms)
    }
    ranked = []
    for (source, text), tokens, length in zip(documents, tokenized, lengths):
        frequencies = {term: tokens.count(term) for term in set(query_terms)}
        score = 0.0
        for term, frequency in frequencies.items():
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log1p(
                (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * length / max(average_length, 1.0)
            )
            score += inverse_frequency * frequency * 2.5 / denominator
        if score > 0:
            ranked.append(RetrievedDocument(source, text.strip(), score))

    return sorted(ranked, key=lambda item: (-item.score, item.source))[:top_k]


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Format retrieved notes with source labels for inclusion in an LLM prompt."""
    results = retrieve(query, top_k=top_k)
    return "\n\n".join(
        f"[{item.source}; BM25={item.score:.3f}]\n{item.text}"
        for item in results
    )
