"""
chunker.py
Splits cleaned page text into chunks using two strategies:
  1. fixed_token  — fixed-size token windows with overlap
  2. sentence_window — groups of N consecutive sentences with overlap

All strategies return a flat list of chunk dicts containing text + position info.
"""

import re
from typing import List, Dict, Any

from src.utils.config_loader import load_chunking_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed — falling back to approximate word-based tokenization")


# ── Token counting helpers ─────────────────────────────────────────────────────

_encoders: Dict[str, Any] = {}


def get_encoder(encoding_name: str = "cl100k_base"):
    if not TIKTOKEN_AVAILABLE:
        return None
    if encoding_name not in _encoders:
        _encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoders[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = get_encoder(encoding_name)
    if enc is None:
        # Approximation: ~0.75 tokens per word
        return int(len(text.split()) * 1.33)
    return len(enc.encode(text))


def encode_text(text: str, encoding_name: str = "cl100k_base") -> List[int]:
    enc = get_encoder(encoding_name)
    if enc is None:
        # Return word indices as proxy
        words = text.split()
        return list(range(len(words)))
    return enc.encode(text)


def decode_tokens(tokens: List[int], encoding_name: str = "cl100k_base") -> str:
    enc = get_encoder(encoding_name)
    if enc is None:
        return " ".join(str(t) for t in tokens)
    return enc.decode(tokens)


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"])|"   # Standard sentence endings
    r"(?<=\n)\s*(?=[A-Z\d])",         # Paragraph breaks
)


def split_sentences(text: str) -> List[str]:
    """Simple regex sentence splitter. Good enough for financial text."""
    sentences = _SENT_SPLIT_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Strategy: fixed_token ─────────────────────────────────────────────────────

def chunk_fixed_token(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    tokenizer: str = "cl100k_base",
    min_chunk_tokens: int = 50,
    respect_sentence_boundaries: bool = True,
) -> List[Dict]:
    """
    Split text into fixed-size token windows with overlap.
    If respect_sentence_boundaries=True, tries to end chunks at sentence
    boundaries within a ±10% tolerance of chunk_size.
    """
    if respect_sentence_boundaries:
        return _chunk_fixed_sentence_aware(
            text, chunk_size, chunk_overlap, tokenizer, min_chunk_tokens
        )

    tokens = encode_text(text, tokenizer)
    total = len(tokens)
    chunks = []
    start = 0
    idx = 0

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]

        if TIKTOKEN_AVAILABLE:
            chunk_text = decode_tokens(chunk_tokens, tokenizer)
        else:
            # Approximate: slice text by word proportion
            words = text.split()
            word_start = int(start / max(total, 1) * len(words))
            word_end = int(end / max(total, 1) * len(words))
            chunk_text = " ".join(words[word_start:word_end])

        token_count = len(chunk_tokens)
        if token_count >= min_chunk_tokens:
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_index": idx,
                "token_count": token_count,
                "start_token": start,
                "end_token": end,
            })
            idx += 1

        if end >= total:
            break
        start = end - chunk_overlap

    return chunks


def _chunk_fixed_sentence_aware(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: str,
    min_chunk_tokens: int,
) -> List[Dict]:
    """Fixed-token chunking that tries to respect sentence boundaries."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sents = []
    current_tokens = 0
    idx = 0
    overlap_buffer = []

    for sent in sentences:
        sent_tokens = count_tokens(sent, tokenizer)

        if current_tokens + sent_tokens > chunk_size and current_sents:
            chunk_text = " ".join(current_sents)
            tc = count_tokens(chunk_text, tokenizer)
            if tc >= min_chunk_tokens:
                chunks.append({
                    "text": chunk_text.strip(),
                    "chunk_index": idx,
                    "token_count": tc,
                    "start_token": -1,   # Not tracked in sentence mode
                    "end_token": -1,
                })
                idx += 1

            # Build overlap: take sentences from end of current chunk
            overlap_tokens = 0
            overlap_sents = []
            for s in reversed(current_sents):
                st = count_tokens(s, tokenizer)
                if overlap_tokens + st > chunk_overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += st

            current_sents = overlap_sents + [sent]
            current_tokens = overlap_tokens + sent_tokens
        else:
            current_sents.append(sent)
            current_tokens += sent_tokens

    # Last chunk
    if current_sents:
        chunk_text = " ".join(current_sents)
        tc = count_tokens(chunk_text, tokenizer)
        if tc >= min_chunk_tokens:
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_index": idx,
                "token_count": tc,
                "start_token": -1,
                "end_token": -1,
            })

    return chunks


# ── Strategy: sentence_window ─────────────────────────────────────────────────

def chunk_sentence_window(
    text: str,
    window_size: int = 5,
    overlap_sentences: int = 1,
    min_chunk_tokens: int = 30,
    tokenizer: str = "cl100k_base",
) -> List[Dict]:
    """Group consecutive sentences into windows."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    step = max(1, window_size - overlap_sentences)
    idx = 0

    for start in range(0, len(sentences), step):
        window = sentences[start: start + window_size]
        chunk_text = " ".join(window)
        tc = count_tokens(chunk_text, tokenizer)
        if tc >= min_chunk_tokens:
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_index": idx,
                "token_count": tc,
                "start_token": -1,
                "end_token": -1,
            })
            idx += 1

    return chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

def chunk_text(text: str, config: dict = None) -> List[Dict]:
    """
    Chunk text according to the given config dict.
    If config is None, loads the 'default' chunking config.
    """
    if config is None:
        config = load_chunking_config("default")

    strategy = config.get("strategy", "fixed_token")

    if strategy == "fixed_token":
        return chunk_fixed_token(
            text,
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 50),
            tokenizer=config.get("tokenizer", "cl100k_base"),
            min_chunk_tokens=config.get("min_chunk_tokens", 50),
            respect_sentence_boundaries=config.get("respect_sentence_boundaries", True),
        )
    elif strategy == "sentence_window":
        return chunk_sentence_window(
            text,
            window_size=config.get("window_size", 5),
            overlap_sentences=config.get("overlap_sentences", 1),
            min_chunk_tokens=config.get("min_chunk_tokens", 30),
            tokenizer=config.get("tokenizer", "cl100k_base"),
        )
    elif strategy == "semantic":
        from src.chunking.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(
            max_chunk_tokens=config.get("max_chunk_tokens", 600),
            min_chunk_tokens=config.get("min_chunk_tokens", 80),
            tokenizer=config.get("tokenizer", "cl100k_base"),
        )
        return chunker.chunk_text(text)
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Use 'fixed_token', 'sentence_window', or 'semantic'"
        )


def chunk_pages(pages: List[dict], config: dict = None) -> List[Dict]:
    """
    Chunk a list of page dicts (from cleaner.clean_pages()).
    Returns flat list of chunk dicts with page_number attached.
    """
    if config is None:
        config = load_chunking_config("default")

    all_chunks = []
    global_idx = 0

    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue

        page_chunks = chunk_text(text, config)
        for c in page_chunks:
            c["page_number"] = page.get("page_number", 0)
            c["global_chunk_index"] = global_idx
            global_idx += 1
            all_chunks.append(c)

    logger.debug(f"Chunking: {len(pages)} pages → {len(all_chunks)} chunks")
    return all_chunks
