"""
context_manager.py
Manages context window to ensure retrieved chunks fit within model token limits.

This module addresses token limit errors (e.g., q015) where multi-period
comparison queries exceeded the model's context window.
"""

import re
from typing import List, Dict, Optional, Tuple


class ContextManager:
    """
    Manage context to fit within model token limits.

    Handles:
    - Token counting (approximate)
    - Dynamic truncation of context
    - Prioritization of high-ranked chunks
    - Graceful degradation for long contexts
    """

    # Approximate tokens per character for English text
    # Using 4 chars per token as rough estimate (works for most models)
    CHARS_PER_TOKEN = 4

    # Model context limits (conservative estimates)
    MODEL_LIMITS = {
        'qwen2.5-7b': 8192,
        'qwen2.5-14b': 8192,
        'qwen2.5-32b': 32768,
        'gpt-4': 8192,
        'gpt-4-turbo': 128000,
        'default': 8192,
    }

    def __init__(
        self,
        model_name: str = "qwen2.5-14b",
        max_context_tokens: Optional[int] = None,
        reserved_for_output: int = 512,
        reserved_for_prompt: int = 800,
    ):
        """
        Initialize ContextManager.

        Args:
            model_name: Name of the model (for automatic limit detection)
            max_context_tokens: Override automatic limit detection
            reserved_for_output: Tokens reserved for model output
            reserved_for_prompt: Tokens reserved for system/user prompt template
        """
        self.model_name = model_name.lower()
        self.reserved_for_output = reserved_for_output
        self.reserved_for_prompt = reserved_for_prompt

        # Determine model limit
        model_limit = self.MODEL_LIMITS.get(
            self.model_name,
            self.MODEL_LIMITS['default']
        )

        # Calculate available tokens for context
        if max_context_tokens:
            self.max_context_tokens = max_context_tokens
        else:
            self.max_context_tokens = model_limit - reserved_for_output - reserved_for_prompt

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses character-based approximation. For production, consider using
        the actual tokenizer (tiktoken for OpenAI, transformers for HuggingFace).
        """
        return len(text) // self.CHARS_PER_TOKEN

    def count_chunk_tokens(self, chunk: Dict) -> int:
        """Count tokens in a chunk including metadata overhead."""
        text = chunk.get('text', '')
        # Add overhead for metadata formatting in prompt
        metadata_overhead = 50  # [Doc-N] headers, fiscal period, etc.
        return self.count_tokens(text) + metadata_overhead

    def fit_context(
        self,
        chunks: List[Dict],
        min_chunks: int = 3,
    ) -> Tuple[List[Dict], Dict]:
        """
        Select chunks that fit within token budget.

        Prioritizes by order (assumes chunks are pre-sorted by relevance/rerank score).

        Args:
            chunks: List of chunk dictionaries, sorted by relevance
            min_chunks: Minimum number of chunks to include (will truncate if needed)

        Returns:
            Tuple of (selected_chunks, stats_dict)
        """
        if not chunks:
            return [], {'original_count': 0, 'selected_count': 0, 'truncated': False}

        available_tokens = self.max_context_tokens
        selected_chunks = []
        current_tokens = 0
        truncation_occurred = False

        for i, chunk in enumerate(chunks):
            chunk_tokens = self.count_chunk_tokens(chunk)

            # Always try to include minimum chunks, with truncation if needed
            if i < min_chunks:
                if current_tokens + chunk_tokens <= available_tokens:
                    selected_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    # Truncate this chunk to fit
                    remaining_tokens = available_tokens - current_tokens
                    if remaining_tokens > 100:  # Minimum useful size
                        truncated_chunk = self._truncate_chunk(chunk, remaining_tokens)
                        selected_chunks.append(truncated_chunk)
                        current_tokens += remaining_tokens
                        truncation_occurred = True
                    break
            else:
                # Beyond minimum, only add if fits completely
                if current_tokens + chunk_tokens <= available_tokens:
                    selected_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    # Try partial fit for one more chunk
                    remaining_tokens = available_tokens - current_tokens
                    if remaining_tokens > 200:  # Worth including partial
                        truncated_chunk = self._truncate_chunk(chunk, remaining_tokens)
                        selected_chunks.append(truncated_chunk)
                        truncation_occurred = True
                    break

        stats = {
            'original_count': len(chunks),
            'selected_count': len(selected_chunks),
            'original_tokens': sum(self.count_chunk_tokens(c) for c in chunks),
            'selected_tokens': current_tokens,
            'max_tokens': self.max_context_tokens,
            'truncated': truncation_occurred,
        }

        return selected_chunks, stats

    def _truncate_chunk(self, chunk: Dict, max_tokens: int) -> Dict:
        """
        Truncate a chunk to fit within token limit.

        Attempts to preserve sentence boundaries and important content.
        """
        text = chunk.get('text', '')
        target_chars = max_tokens * self.CHARS_PER_TOKEN

        if len(text) <= target_chars:
            return chunk

        truncated_text = text[:target_chars]

        # Try to end at a sentence boundary
        # Look for last sentence-ending punctuation
        last_period = max(
            truncated_text.rfind('. '),
            truncated_text.rfind('.\n'),
            truncated_text.rfind('? '),
            truncated_text.rfind('! '),
        )

        # Only use sentence boundary if we keep at least 70% of truncated text
        if last_period > len(truncated_text) * 0.7:
            truncated_text = truncated_text[:last_period + 1]
        else:
            # Fall back to word boundary
            last_space = truncated_text.rfind(' ')
            if last_space > len(truncated_text) * 0.9:
                truncated_text = truncated_text[:last_space]

        truncated_text = truncated_text.strip() + " [...]"

        return {
            **chunk,
            'text': truncated_text,
            'truncated': True,
            'original_length': len(text),
        }

    def format_context_for_prompt(
        self,
        chunks: List[Dict],
        include_metadata: bool = True,
    ) -> str:
        """
        Format chunks into a context string for the prompt.

        Args:
            chunks: List of chunk dictionaries
            include_metadata: Whether to include fiscal period metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})

            # Build header
            header_parts = [f"[Doc-{i}]"]

            if include_metadata:
                fiscal_period = metadata.get('fiscal_period', '')
                doc_type = metadata.get('doc_type', '')

                if fiscal_period:
                    header_parts.append(f"({fiscal_period}")
                    if doc_type:
                        header_parts[-1] += f", {doc_type})"
                    else:
                        header_parts[-1] += ")"

            header = ' '.join(header_parts)

            # Add truncation indicator
            if chunk.get('truncated'):
                header += " [TRUNCATED]"

            context_parts.append(f"{header}\n{text}")

        return "\n\n".join(context_parts)


class ContextOptimizer:
    """
    Advanced context optimization strategies.
    """

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def optimize_for_multi_period(
        self,
        chunks: List[Dict],
        periods_required: List[str],
    ) -> List[Dict]:
        """
        Optimize context selection for multi-period comparison queries.

        Ensures balanced representation of each required period.

        Args:
            chunks: All retrieved chunks
            periods_required: List of fiscal periods that should be represented

        Returns:
            Balanced selection of chunks
        """
        # Group chunks by fiscal period
        period_chunks = {}
        other_chunks = []

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            period = metadata.get('fiscal_period', '')

            matched_period = None
            for required in periods_required:
                if required in period:
                    matched_period = required
                    break

            if matched_period:
                if matched_period not in period_chunks:
                    period_chunks[matched_period] = []
                period_chunks[matched_period].append(chunk)
            else:
                other_chunks.append(chunk)

        # Calculate fair allocation per period
        available_tokens = self.context_manager.max_context_tokens
        n_periods = len(periods_required)
        tokens_per_period = available_tokens // max(n_periods, 1)

        # Select chunks from each period
        balanced_chunks = []
        for period in periods_required:
            period_selection = period_chunks.get(period, [])
            current_tokens = 0

            for chunk in period_selection:
                chunk_tokens = self.context_manager.count_chunk_tokens(chunk)
                if current_tokens + chunk_tokens <= tokens_per_period:
                    balanced_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    break

        # Fill remaining space with other relevant chunks
        remaining_tokens = available_tokens - sum(
            self.context_manager.count_chunk_tokens(c) for c in balanced_chunks
        )

        for chunk in other_chunks:
            chunk_tokens = self.context_manager.count_chunk_tokens(chunk)
            if chunk_tokens <= remaining_tokens:
                balanced_chunks.append(chunk)
                remaining_tokens -= chunk_tokens

        return balanced_chunks

    def extract_key_sentences(
        self,
        text: str,
        max_sentences: int = 5,
    ) -> str:
        """
        Extract key sentences from text based on financial content.

        Useful for aggressive compression when context is very limited.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= max_sentences:
            return text

        # Score sentences by financial content
        scored = []
        for sent in sentences:
            score = 0

            # Numbers are important
            numbers = re.findall(r'\$?[\d,]+\.?\d*\s*(billion|million|%)?', sent, re.I)
            score += len(numbers) * 3

            # Financial terms
            financial_terms = [
                'revenue', 'income', 'growth', 'margin', 'segment',
                'quarter', 'fiscal', 'increased', 'decreased', 'profit',
                'loss', 'earnings', 'operating', 'net', 'gross', 'total'
            ]
            for term in financial_terms:
                if term in sent.lower():
                    score += 1

            # Year references
            if re.search(r'FY\d{4}|20\d{2}', sent):
                score += 2

            scored.append((score, sent))

        # Select top sentences
        scored.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored[:max_sentences]]

        # Restore original order
        ordered = [s for s in sentences if s in top_sentences]

        return ' '.join(ordered)
