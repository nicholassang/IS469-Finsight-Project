# FinSight LLM Improvement Recommendations
## Technical Report: Enhancing RAG System Performance

**Date:** March 2026
**System:** FinSight v1.0
**Focus:** Addressing identified failure modes and improving overall accuracy

---

## Executive Summary

Based on comprehensive evaluation of the FinSight RAG system, we identified **4 distinct failure modes** affecting 20% of queries. This report provides actionable recommendations to address each failure mode, with implementation priorities and expected impact.

### Current Performance Baseline

| Metric | Baseline | Advanced | Target |
|--------|----------|----------|--------|
| Faithfulness | 0.607 | 0.839 | **0.92+** |
| Answer Relevancy | 0.485 | 0.904 | **0.95+** |
| Context Recall | 0.225 | 0.733 | **0.85+** |
| Context Precision | 0.456 | 0.526 | **0.70+** |
| Error Rate | ~35% | ~20% | **<10%** |

---

## Part 1: Addressing Identified Failure Modes

### 1.1 Temporal Query Handling (CRITICAL)

**Problem:** The system struggles with quarter-specific queries, retrieving data from wrong fiscal periods.

**Affected Questions:** q007, q010, q015

**Root Cause:**
- Embedding models treat "Q1 FY2025" and "Q2 FY2025" as semantically similar
- No explicit temporal filtering before semantic search
- Fiscal period metadata underutilized in retrieval

#### Recommended Solutions:

**Solution A: Query-Time Fiscal Period Extraction (High Priority)**

```python
# Add to src/retrieval/query_processor.py

import re
from typing import Optional, List, Dict

class FiscalPeriodExtractor:
    """Extract and normalize fiscal period references from queries."""

    FISCAL_PATTERNS = [
        # FY2024, FY2025, etc.
        r'FY\s*(\d{4})',
        # Q1 FY2024, Q2 FY2025, etc.
        r'Q([1-4])\s*FY\s*(\d{4})',
        # Q1 2024, Q2 2025 (calendar year)
        r'Q([1-4])\s+(\d{4})',
        # fiscal year 2024
        r'fiscal\s+year\s+(\d{4})',
        # first quarter 2024
        r'(first|second|third|fourth)\s+quarter\s+(\d{4})',
    ]

    QUARTER_MAP = {
        'first': '1', 'second': '2', 'third': '3', 'fourth': '4'
    }

    def extract(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract fiscal period from query.
        Returns: {'fiscal_year': 'FY2024', 'quarter': 'Q1', 'raw': 'Q1 FY2024'}
        """
        query_lower = query.lower()
        result = {'fiscal_year': None, 'quarter': None, 'raw': None}

        # Try Q# FY#### pattern first (most specific)
        match = re.search(r'q([1-4])\s*fy\s*(\d{4})', query_lower)
        if match:
            result['quarter'] = f"Q{match.group(1)}"
            result['fiscal_year'] = f"FY{match.group(2)}"
            result['raw'] = f"Q{match.group(1)} FY{match.group(2)}"
            return result

        # Try FY#### pattern
        match = re.search(r'fy\s*(\d{4})', query_lower)
        if match:
            result['fiscal_year'] = f"FY{match.group(1)}"
            result['raw'] = f"FY{match.group(1)}"
            return result

        # Try written quarters
        for word, num in self.QUARTER_MAP.items():
            if word in query_lower:
                result['quarter'] = f"Q{num}"
                # Look for associated year
                year_match = re.search(r'(\d{4})', query)
                if year_match:
                    result['fiscal_year'] = f"FY{year_match.group(1)}"
                    result['raw'] = f"Q{num} FY{year_match.group(1)}"
                break

        return result

    def to_metadata_filter(self, extracted: Dict) -> Optional[Dict]:
        """Convert extracted period to ChromaDB metadata filter."""
        if extracted['raw']:
            return {"fiscal_period": {"$eq": extracted['raw']}}
        elif extracted['fiscal_year']:
            return {"fiscal_period": {"$contains": extracted['fiscal_year']}}
        return None
```

**Solution B: Metadata Pre-Filtering in Retrieval (High Priority)**

```python
# Modify src/retrieval/dense_retriever.py

class DenseRetriever:
    def retrieve(self, query: str, top_k: int = 10,
                 fiscal_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve with optional fiscal period filtering.
        """
        # Extract fiscal period from query
        extractor = FiscalPeriodExtractor()
        fiscal_info = extractor.extract(query)

        # Build where clause for ChromaDB
        where_clause = fiscal_filter or extractor.to_metadata_filter(fiscal_info)

        # Query with filter
        if where_clause:
            results = self.collection.query(
                query_embeddings=[self._embed(query)],
                n_results=top_k * 2,  # Over-retrieve then filter
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_embeddings=[self._embed(query)],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        return self._format_results(results)[:top_k]
```

**Solution C: Temporal-Aware Query Expansion (Medium Priority)**

```python
# Add to src/retrieval/query_expander.py

class TemporalQueryExpander:
    """Expand queries with explicit temporal context."""

    MSFT_FISCAL_CALENDAR = {
        'Q1': 'July-September',
        'Q2': 'October-December',
        'Q3': 'January-March',
        'Q4': 'April-June'
    }

    def expand(self, query: str, fiscal_info: Dict) -> str:
        """
        Expand query with temporal context to improve retrieval.

        Example:
        Input: "What was revenue in Q2 FY2025?"
        Output: "What was revenue in Q2 FY2025 October-December 2024 quarter
                 ended December 31 2024 10-Q"
        """
        expansions = []

        if fiscal_info.get('quarter') and fiscal_info.get('fiscal_year'):
            q = fiscal_info['quarter']
            fy = fiscal_info['fiscal_year']

            # Add calendar month range
            month_range = self.MSFT_FISCAL_CALENDAR.get(q, '')
            if month_range:
                expansions.append(month_range)

            # Add quarter end date
            fy_num = int(fy.replace('FY', ''))
            quarter_ends = {
                'Q1': f"September 30 {fy_num - 1}",
                'Q2': f"December 31 {fy_num - 1}",
                'Q3': f"March 31 {fy_num}",
                'Q4': f"June 30 {fy_num}"
            }
            if q in quarter_ends:
                expansions.append(f"quarter ended {quarter_ends[q]}")

            # Add document type hint
            if q in ['Q1', 'Q2', 'Q3']:
                expansions.append("10-Q quarterly report")
            else:
                expansions.append("10-K annual report")

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query
```

**Expected Impact:**
- Reduce temporal retrieval failures by 80%
- Improve context precision for quarterly queries by 25%

---

### 1.2 Context Window Management (HIGH)

**Problem:** Multi-period comparison queries exceed token limits (8192 tokens for Qwen2.5-14B).

**Affected Questions:** q015

**Root Cause:**
- 12 retrieved chunks × ~600 tokens = ~7200 tokens
- Plus prompt template = exceeds limit
- No dynamic truncation

#### Recommended Solutions:

**Solution A: Dynamic Context Truncation (High Priority)**

```python
# Add to src/generation/context_manager.py

import tiktoken
from typing import List, Dict

class ContextManager:
    """Manage context to fit within model token limits."""

    def __init__(self, model_name: str = "qwen2.5-14b",
                 max_context_tokens: int = 6000,
                 reserved_for_output: int = 512):
        self.max_context_tokens = max_context_tokens
        self.reserved_for_output = reserved_for_output
        # Use cl100k_base as approximation for Qwen tokenization
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def fit_context(self, chunks: List[Dict],
                    prompt_template_tokens: int = 500) -> List[Dict]:
        """
        Select chunks that fit within token budget.
        Prioritizes by rerank score (assumes chunks are pre-sorted).
        """
        available_tokens = self.max_context_tokens - prompt_template_tokens

        selected_chunks = []
        current_tokens = 0

        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = self.count_tokens(chunk_text)

            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to include truncated version of important chunks
                remaining = available_tokens - current_tokens
                if remaining > 200:  # Minimum useful chunk size
                    truncated_text = self._truncate_to_tokens(chunk_text, remaining)
                    truncated_chunk = {**chunk, 'text': truncated_text, 'truncated': True}
                    selected_chunks.append(truncated_chunk)
                break

        return selected_chunks

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit, preserving sentence boundaries."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)

        # Try to end at sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.7:
            return truncated_text[:last_period + 1] + " [truncated]"

        return truncated_text + "... [truncated]"
```

**Solution B: Context Compression via Summarization (Medium Priority)**

```python
# Add to src/generation/context_compressor.py

class ContextCompressor:
    """Compress context when it exceeds limits."""

    COMPRESSION_PROMPT = """Summarize the following financial document excerpt,
preserving all numerical values, dates, and key metrics:

{text}

Concise summary (keep all numbers):"""

    def compress_if_needed(self, chunks: List[Dict],
                           max_tokens: int = 6000) -> List[Dict]:
        """
        Compress chunks if total exceeds max_tokens.
        Uses extractive summarization to preserve key financial data.
        """
        total_tokens = sum(self.count_tokens(c['text']) for c in chunks)

        if total_tokens <= max_tokens:
            return chunks

        # Strategy: Keep top-ranked chunks intact, compress lower-ranked ones
        compressed = []
        budget = max_tokens

        for i, chunk in enumerate(chunks):
            chunk_tokens = self.count_tokens(chunk['text'])

            if i < 3:  # Keep top 3 chunks intact
                compressed.append(chunk)
                budget -= chunk_tokens
            elif budget > 200:
                # Compress this chunk
                compression_ratio = min(budget / chunk_tokens, 0.5)
                summary = self._extractive_summary(chunk['text'], compression_ratio)
                compressed.append({**chunk, 'text': summary, 'compressed': True})
                budget -= self.count_tokens(summary)

        return compressed

    def _extractive_summary(self, text: str, ratio: float) -> str:
        """Extract key sentences containing numbers and financial terms."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Score sentences by financial content
        scored = []
        for sent in sentences:
            score = 0
            # Numbers are important
            score += len(re.findall(r'\$?[\d,]+\.?\d*\s*(billion|million|%)?', sent)) * 3
            # Financial terms
            financial_terms = ['revenue', 'income', 'growth', 'margin', 'segment',
                             'quarter', 'fiscal', 'increased', 'decreased']
            score += sum(1 for term in financial_terms if term in sent.lower())
            scored.append((score, sent))

        # Select top sentences by score
        scored.sort(reverse=True)
        n_keep = max(1, int(len(sentences) * ratio))
        kept = [s[1] for s in scored[:n_keep]]

        # Restore original order
        result = [s for s in sentences if s in kept]
        return ' '.join(result)
```

**Solution C: Upgrade to Larger Context Model (Long-term)**

```yaml
# Update config/settings.yaml

generation:
  backend: openai
  model: qwen2.5-32b  # Upgrade from 14B
  # OR use a long-context model
  # model: qwen2.5-14b-32k  # If available
  max_model_len: 32768
  temperature: 0.0
  max_tokens: 1024
```

**Expected Impact:**
- Eliminate token limit errors completely
- Enable complex multi-document reasoning queries

---

### 1.3 Retrieval Precision Enhancement (MEDIUM)

**Problem:** System sometimes retrieves topically related but factually incorrect chunks.

**Affected Questions:** q007 (wrong quarter), q005 (indirect retrieval)

#### Recommended Solutions:

**Solution A: Multi-Stage Retrieval with Verification (High Priority)**

```python
# Add to src/retrieval/verified_retriever.py

class VerifiedRetriever:
    """
    Multi-stage retrieval with fact verification.
    Stage 1: Broad retrieval (top-50)
    Stage 2: Metadata filtering
    Stage 3: Cross-encoder reranking
    Stage 4: Fact verification (optional)
    """

    def __init__(self, cfg: dict):
        self.dense = DenseRetriever(cfg)
        self.sparse = SparseRetriever(cfg)
        self.reranker = Reranker(cfg)
        self.extractor = FiscalPeriodExtractor()

    def retrieve(self, query: str, final_k: int = 10) -> List[Dict]:
        # Stage 1: Broad retrieval
        dense_results = self.dense.retrieve(query, top_k=50)
        sparse_results = self.sparse.retrieve(query, top_k=50)

        # Stage 2: Metadata filtering
        fiscal_info = self.extractor.extract(query)
        if fiscal_info['raw']:
            dense_results = self._filter_by_period(dense_results, fiscal_info)
            sparse_results = self._filter_by_period(sparse_results, fiscal_info)

        # Stage 3: RRF Fusion
        fused = self._rrf_fuse(dense_results, sparse_results)

        # Stage 4: Reranking
        reranked = self.reranker.rerank(query, fused[:30])

        return reranked[:final_k]

    def _filter_by_period(self, chunks: List[Dict], fiscal_info: Dict) -> List[Dict]:
        """Filter chunks to match fiscal period."""
        target = fiscal_info['raw']
        fy = fiscal_info.get('fiscal_year', '')

        filtered = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            chunk_period = metadata.get('fiscal_period', '')

            # Exact match preferred
            if target and target in chunk_period:
                chunk['period_match'] = 'exact'
                filtered.append(chunk)
            # Fiscal year match acceptable
            elif fy and fy in chunk_period:
                chunk['period_match'] = 'year'
                filtered.append(chunk)

        # If too few results, fall back to unfiltered
        if len(filtered) < 5:
            return chunks[:30]

        return filtered
```

**Solution B: Answer Verification Loop (Medium Priority)**

```python
# Add to src/generation/verified_generator.py

class VerifiedGenerator:
    """
    Generate answers with self-verification loop.
    If answer contains numbers, verify they appear in context.
    """

    def generate_with_verification(self, question: str,
                                   context: List[Dict]) -> Dict:
        # Generate initial answer
        answer = self.generator.generate(question, context)

        # Extract numbers from answer
        answer_numbers = self._extract_numbers(answer['answer'])

        # Verify each number appears in context
        context_text = ' '.join(c['text'] for c in context)
        context_numbers = self._extract_numbers(context_text)

        unverified = []
        for num in answer_numbers:
            if not self._number_in_context(num, context_numbers):
                unverified.append(num)

        if unverified:
            # Re-generate with stricter prompt
            answer = self._regenerate_strict(question, context, unverified)
            answer['verification_triggered'] = True
            answer['unverified_numbers'] = unverified

        return answer

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract financial numbers from text."""
        import re
        patterns = [
            r'\$[\d,]+\.?\d*\s*(billion|million|thousand)?',
            r'[\d,]+\.?\d*\s*(billion|million|thousand)',
            r'[\d,]+\.?\d*%',
            r'\d{1,3}(,\d{3})+',
        ]
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text, re.IGNORECASE))
        return numbers

    def _regenerate_strict(self, question: str, context: List[Dict],
                          unverified: List[str]) -> Dict:
        """Re-generate with explicit warning about unverified numbers."""
        strict_prompt = f"""
WARNING: Your previous answer contained numbers that were not found in the context: {unverified}

CRITICAL: Only use numbers that EXACTLY appear in the provided context.
If the exact number is not in the context, say "The specific figure is not stated in the provided excerpts."

{question}
"""
        return self.generator.generate(strict_prompt, context)
```

**Expected Impact:**
- Reduce wrong-data retrieval by 60%
- Improve faithfulness to 0.90+

---

### 1.4 Chunk Quality Improvement (MEDIUM)

**Problem:** Fixed-size chunking splits important information across boundaries.

**Affected Questions:** q005 (had to derive from segments), tables split

#### Recommended Solutions:

**Solution A: Semantic Chunking (High Priority)**

```python
# Add to src/chunking/semantic_chunker.py

class SemanticChunker:
    """
    Chunk by semantic boundaries rather than fixed token counts.
    Preserves section headers, tables, and logical units.
    """

    SEC_SECTION_PATTERNS = [
        r'^PART\s+[IV]+',
        r'^ITEM\s+\d+[A-Z]?\.?',
        r'^(RISK FACTORS|MANAGEMENT.S DISCUSSION|FINANCIAL STATEMENTS)',
        r'^(Notes to|Note \d+)',
    ]

    def chunk_document(self, text: str, doc_config: dict,
                       max_chunk_size: int = 800,
                       min_chunk_size: int = 200) -> List[Dict]:
        """
        Create semantically coherent chunks.
        """
        # Step 1: Identify section boundaries
        sections = self._split_by_sections(text)

        chunks = []
        for section in sections:
            # Step 2: Further split large sections by paragraphs
            if self._count_tokens(section['text']) > max_chunk_size:
                sub_chunks = self._split_section(section, max_chunk_size, min_chunk_size)
                chunks.extend(sub_chunks)
            elif self._count_tokens(section['text']) >= min_chunk_size:
                chunks.append(section)
            else:
                # Merge small sections with previous
                if chunks:
                    chunks[-1]['text'] += '\n\n' + section['text']
                else:
                    chunks.append(section)

        return chunks

    def _split_by_sections(self, text: str) -> List[Dict]:
        """Split document by SEC filing section headers."""
        import re

        combined_pattern = '|'.join(f'({p})' for p in self.SEC_SECTION_PATTERNS)

        sections = []
        last_end = 0
        last_header = "Introduction"

        for match in re.finditer(combined_pattern, text, re.MULTILINE | re.IGNORECASE):
            if match.start() > last_end:
                sections.append({
                    'text': text[last_end:match.start()].strip(),
                    'section_title': last_header
                })
            last_header = match.group(0).strip()
            last_end = match.start()

        # Add final section
        if last_end < len(text):
            sections.append({
                'text': text[last_end:].strip(),
                'section_title': last_header
            })

        return sections
```

**Solution B: Table-Aware Chunking (High Priority)**

```python
# Add to src/chunking/table_chunker.py

class TableAwareChunker:
    """
    Detect and preserve tables as atomic chunks.
    """

    def extract_tables(self, text: str) -> List[Dict]:
        """
        Identify tabular data and extract as separate chunks.
        """
        import re

        tables = []

        # Pattern for detecting table-like structures
        # Multiple aligned columns with numbers
        table_pattern = r'((?:^.{0,50}\t.+$\n?){3,})'

        for match in re.finditer(table_pattern, text, re.MULTILINE):
            table_text = match.group(0)

            # Verify it's actually a table (has consistent column structure)
            if self._is_valid_table(table_text):
                tables.append({
                    'text': table_text,
                    'type': 'table',
                    'start': match.start(),
                    'end': match.end()
                })

        return tables

    def chunk_with_tables(self, text: str, doc_config: dict) -> List[Dict]:
        """
        Chunk text while preserving tables as atomic units.
        """
        tables = self.extract_tables(text)

        # Remove tables from text, replace with markers
        modified_text = text
        for i, table in enumerate(reversed(tables)):
            marker = f"[TABLE_{i}]"
            modified_text = modified_text[:table['start']] + marker + modified_text[table['end']:]

        # Chunk the remaining text normally
        from src.chunking.chunker import chunk_pages
        chunks = chunk_pages([{'text': modified_text}], doc_config)

        # Re-insert tables as separate chunks
        final_chunks = []
        for chunk in chunks:
            if '[TABLE_' in chunk['text']:
                # Split around table markers
                parts = re.split(r'\[TABLE_(\d+)\]', chunk['text'])
                for j, part in enumerate(parts):
                    if part.isdigit():
                        # This is a table index
                        table_chunk = {
                            **chunk,
                            'text': tables[int(part)]['text'],
                            'is_table': True
                        }
                        final_chunks.append(table_chunk)
                    elif part.strip():
                        final_chunks.append({**chunk, 'text': part.strip()})
            else:
                final_chunks.append(chunk)

        return final_chunks
```

**Expected Impact:**
- Reduce chunking-related errors by 70%
- Improve context quality for table-based queries

---

## Part 2: Model-Level Improvements

### 2.1 Fine-Tuning for Financial Domain

**Option A: Embedding Model Fine-Tuning**

```python
# Fine-tune embedding model on financial QA pairs
# Improves semantic understanding of fiscal terminology

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def fine_tune_embedder():
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Create training pairs from your evaluation data
    train_examples = [
        # Positive pairs: question + correct context
        InputExample(texts=[
            "What was Microsoft's total revenue for FY2024?",
            "Total revenue was $245,122 million for the fiscal year ended June 30, 2024"
        ], label=1.0),

        # Hard negatives: question + wrong quarter (but similar topic)
        InputExample(texts=[
            "What was Microsoft's revenue in Q2 FY2025?",
            "Revenue for Q1 FY2025 was $65,585 million"  # Wrong quarter!
        ], label=0.2),  # Low similarity score
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path='models/finsight-mpnet-finetuned'
    )
```

**Option B: Reranker Fine-Tuning**

```python
# Fine-tune cross-encoder on financial relevance judgments

from sentence_transformers import CrossEncoder

def fine_tune_reranker():
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Training data: (query, passage, relevance_score)
    train_samples = [
        # Highly relevant (same fiscal period, direct answer)
        ("Q2 FY2025 revenue", "Revenue for Q2 FY2025 was $69.6 billion", 1.0),

        # Partially relevant (right company, wrong period)
        ("Q2 FY2025 revenue", "Revenue for Q1 FY2025 was $65.6 billion", 0.3),

        # Irrelevant (different topic)
        ("Q2 FY2025 revenue", "Microsoft's risk factors include...", 0.0),
    ]

    model.fit(
        train_samples,
        epochs=3,
        evaluation_steps=100,
        output_path='models/finsight-reranker-finetuned'
    )
```

### 2.2 Prompt Engineering Improvements

```yaml
# Update config/prompts.yaml

qa_system: |
  You are FinSight, a precise financial disclosure assistant for Microsoft Corporation.

  CRITICAL INSTRUCTION — TEMPORAL ACCURACY:
  When the question mentions a specific fiscal period (e.g., "Q2 FY2025"), you MUST:
  1. First identify which document excerpts are from that EXACT period
  2. Only use data from matching periods
  3. If no excerpt matches the requested period, say: "The provided context does not
     contain data specifically for [requested period]."

  MICROSOFT FISCAL CALENDAR REMINDER:
  - Fiscal Year runs July 1 – June 30
  - Q1 = July-September, Q2 = October-December, Q3 = January-March, Q4 = April-June
  - FY2025 Q2 = October-December 2024 (quarter ended December 31, 2024)

  NUMBER VERIFICATION:
  Before including any number in your answer:
  1. Locate the exact number in the context excerpts
  2. Verify the period matches what was asked
  3. If the exact number isn't in context, do not guess or extrapolate

  [Rest of prompt...]
```

---

## Part 3: Infrastructure Improvements

### 3.1 Upgrade to Larger Model

**Current:** Qwen2.5-14B (8K context, ~28GB VRAM)

**Recommended:** Qwen2.5-32B (32K context, ~65GB VRAM on A100-80GB)

Benefits:
- Longer context window eliminates truncation issues
- Better reasoning for multi-hop queries
- Improved factual accuracy

```bash
# Update serve_vllm.sh for A100-80GB
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
MAX_MODEL_LEN=32768
```

### 3.2 Caching Layer

```python
# Add to src/utils/cache.py

import hashlib
import json
from pathlib import Path
from typing import Optional

class QueryCache:
    """Cache retrieval and generation results."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _hash_query(self, query: str, mode: str) -> str:
        return hashlib.md5(f"{query}:{mode}".encode()).hexdigest()

    def get(self, query: str, mode: str) -> Optional[dict]:
        cache_file = self.cache_dir / f"{self._hash_query(query, mode)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def set(self, query: str, mode: str, result: dict, ttl_hours: int = 24):
        cache_file = self.cache_dir / f"{self._hash_query(query, mode)}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'result': result,
                'cached_at': datetime.now().isoformat(),
                'ttl_hours': ttl_hours
            }, f)
```

---

## Part 4: Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

| Task | Impact | Effort |
|------|--------|--------|
| Add fiscal period extractor | HIGH | Low |
| Implement metadata filtering | HIGH | Low |
| Add context truncation | HIGH | Low |
| Update prompts for temporal awareness | MEDIUM | Low |

### Phase 2: Core Improvements (1 week)

| Task | Impact | Effort |
|------|--------|--------|
| Implement verified retriever | HIGH | Medium |
| Add table-aware chunking | MEDIUM | Medium |
| Create answer verification loop | MEDIUM | Medium |
| Set up query caching | LOW | Low |

### Phase 3: Advanced Enhancements (2-4 weeks)

| Task | Impact | Effort |
|------|--------|--------|
| Fine-tune embedding model | HIGH | High |
| Fine-tune reranker | MEDIUM | Medium |
| Implement semantic chunking | MEDIUM | High |
| Upgrade to 32B model | HIGH | Medium |

---

## Part 5: Expected Results After Improvements

### Target Metrics

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Faithfulness | 0.839 | 0.88 | 0.92 | **0.95** |
| Answer Relevancy | 0.904 | 0.93 | 0.95 | **0.97** |
| Context Recall | 0.733 | 0.80 | 0.85 | **0.90** |
| Context Precision | 0.526 | 0.60 | 0.68 | **0.75** |
| Error Rate | 20% | 12% | 8% | **<5%** |

### Category-Specific Improvements

| Category | Current Success | After Improvements |
|----------|-----------------|-------------------|
| Annual Financials | 100% | 100% |
| Quarterly Results | 80% | **95%** (temporal filtering) |
| Multi-Period Comparison | 80% | **95%** (context management) |
| Business Segments | 100% | 100% |

---

## Conclusion

The identified failure modes are addressable with targeted improvements:

1. **Temporal query handling** → Fiscal period extraction + metadata filtering
2. **Context limits** → Dynamic truncation + model upgrade
3. **Retrieval precision** → Multi-stage verification + fine-tuning
4. **Chunk quality** → Semantic + table-aware chunking

Implementing Phase 1 improvements alone should reduce the error rate from 20% to ~12%, with full implementation targeting <5% error rate.

---

*Report generated for FinSight v1.0 improvement planning*
