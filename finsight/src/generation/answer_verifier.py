"""
answer_verifier.py
Post-generation verification to ensure answer quality and grounding.

Phase 2 Enhancement: Checks generated answers for:
1. Proper citation usage ([Doc-N] format)
2. Temporal consistency (mentioned periods match cited sources)
3. Hallucination detection (numbers not in context)
4. Insufficient evidence recognition

Can trigger re-generation with modified prompts if issues detected.
"""

import re
from typing import Dict, List, Optional, Tuple, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerVerifier:
    """
    Verify generated answers for quality and grounding.

    Checks:
    - Citation presence and format
    - Temporal accuracy
    - Number grounding
    - Confidence indicators
    """

    # Patterns for extracting citations
    CITATION_PATTERN = r'\[Doc-(\d+)\]'
    CITATION_ALT_PATTERN = r'\[Source:\s*([^\]]+)\]'

    # Patterns for extracting numbers
    NUMBER_PATTERN = r'\$?([\d,]+\.?\d*)\s*(billion|million|thousand|%)?'

    # Fiscal period patterns
    FISCAL_PATTERN = r'(Q[1-4]\s+)?FY\d{4}|Q[1-4]\s+\d{4}'

    def __init__(self):
        self.verification_rules = [
            self._check_citation_presence,
            self._check_temporal_consistency,
            self._check_number_grounding,
        ]

    def verify(
        self,
        answer: str,
        context: str,
        chunks: List[Dict],
        requested_period: Optional[str] = None,
    ) -> Dict:
        """
        Verify an answer against context and chunks.

        Args:
            answer: Generated answer text
            context: Formatted context string provided to LLM
            chunks: Original chunk dictionaries
            requested_period: Optional fiscal period from query

        Returns:
            Verification results dict with scores and issues
        """
        results = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "warnings": [],
            "citation_count": 0,
            "cited_docs": [],
            "mentioned_periods": [],
        }

        # Run all verification checks
        for check_fn in self.verification_rules:
            check_result = check_fn(answer, context, chunks, requested_period)
            self._merge_results(results, check_result)

        # Calculate overall confidence
        issue_penalty = len(results["issues"]) * 0.2
        warning_penalty = len(results["warnings"]) * 0.1
        results["confidence"] = max(0.0, 1.0 - issue_penalty - warning_penalty)

        # Mark invalid if critical issues found
        if any("CRITICAL" in issue for issue in results["issues"]):
            results["is_valid"] = False

        return results

    def _check_citation_presence(
        self,
        answer: str,
        context: str,
        chunks: List[Dict],
        requested_period: Optional[str],
    ) -> Dict:
        """Check if answer contains proper citations."""
        result = {"issues": [], "warnings": [], "citation_count": 0, "cited_docs": []}

        # Find all citations
        citations = re.findall(self.CITATION_PATTERN, answer)
        alt_citations = re.findall(self.CITATION_ALT_PATTERN, answer)

        result["citation_count"] = len(citations) + len(alt_citations)
        result["cited_docs"] = [int(c) for c in citations]

        # Check citation validity
        num_chunks = len(chunks)
        for doc_num in result["cited_docs"]:
            if doc_num > num_chunks:
                result["issues"].append(
                    f"CRITICAL: Citation [Doc-{doc_num}] references non-existent document "
                    f"(only {num_chunks} documents provided)"
                )

        # Warn if no citations (unless answer is a refusal/insufficient evidence)
        refusal_phrases = [
            "insufficient evidence",
            "do not contain",
            "cannot provide",
            "not available",
            "no information",
        ]
        is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)

        if result["citation_count"] == 0 and not is_refusal:
            result["warnings"].append(
                "No citations found in answer. Consider adding [Doc-N] references."
            )

        return result

    def _check_temporal_consistency(
        self,
        answer: str,
        context: str,
        chunks: List[Dict],
        requested_period: Optional[str],
    ) -> Dict:
        """Check if mentioned fiscal periods match cited sources."""
        result = {"issues": [], "warnings": [], "mentioned_periods": []}

        # Extract fiscal periods mentioned in answer
        mentioned = re.findall(self.FISCAL_PATTERN, answer, re.IGNORECASE)
        result["mentioned_periods"] = list(set(mentioned))

        # If specific period requested, check it's mentioned
        if requested_period:
            period_mentioned = any(
                requested_period.lower() in m.lower()
                for m in mentioned
            )
            if not period_mentioned and mentioned:
                result["warnings"].append(
                    f"Query asked about {requested_period} but answer mentions "
                    f"different periods: {mentioned}"
                )

        # Check cited documents match mentioned periods
        citations = re.findall(self.CITATION_PATTERN, answer)
        for doc_num_str in citations:
            doc_num = int(doc_num_str)
            if doc_num <= len(chunks):
                chunk = chunks[doc_num - 1]
                chunk_period = chunk.get("metadata", {}).get("fiscal_period", "")

                # Check if cited chunk period is mentioned in answer
                if chunk_period and result["mentioned_periods"]:
                    period_match = any(
                        chunk_period.lower() in m.lower() or m.lower() in chunk_period.lower()
                        for m in result["mentioned_periods"]
                    )
                    if not period_match:
                        result["warnings"].append(
                            f"[Doc-{doc_num}] is from {chunk_period} but answer "
                            f"discusses {result['mentioned_periods']}"
                        )

        return result

    def _check_number_grounding(
        self,
        answer: str,
        context: str,
        chunks: List[Dict],
        requested_period: Optional[str],
    ) -> Dict:
        """Check if numbers in answer appear in context."""
        result = {"issues": [], "warnings": []}

        # Extract numbers from answer
        answer_numbers = self._extract_numbers(answer)

        # Extract numbers from context
        context_numbers = self._extract_numbers(context)

        # Check each significant number in answer
        for num, unit in answer_numbers:
            # Skip small numbers (likely not financial data)
            try:
                value = float(num.replace(",", ""))
                if value < 100 and unit not in ["billion", "million", "%"]:
                    continue
            except ValueError:
                continue

            # Check if this number appears in context
            normalized_num = num.replace(",", "")
            found_in_context = any(
                normalized_num in ctx_num.replace(",", "")
                or ctx_num.replace(",", "") in normalized_num
                for ctx_num, _ in context_numbers
            )

            if not found_in_context:
                result["warnings"].append(
                    f"Number '{num} {unit or ''}' in answer not found in context. "
                    f"Possible hallucination."
                )

        return result

    def _extract_numbers(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Extract numbers and their units from text."""
        numbers = []
        for match in re.finditer(self.NUMBER_PATTERN, text, re.IGNORECASE):
            num = match.group(1)
            unit = match.group(2)
            numbers.append((num, unit.lower() if unit else None))
        return numbers

    def _merge_results(self, main: Dict, addition: Dict):
        """Merge verification results."""
        for key in ["issues", "warnings", "mentioned_periods", "cited_docs"]:
            if key in addition:
                main.setdefault(key, []).extend(addition.get(key, []))
        if "citation_count" in addition:
            main["citation_count"] = addition["citation_count"]


class AnswerRefiner:
    """
    Refine answers based on verification results.

    Can suggest prompt modifications or trigger re-generation.
    """

    def __init__(self, generator=None):
        self.generator = generator
        self.verifier = AnswerVerifier()

    def generate_with_verification(
        self,
        question: str,
        chunks: List[Dict],
        max_retries: int = 1,
    ) -> Dict:
        """
        Generate answer and verify, with optional retry.

        Args:
            question: User question
            chunks: Retrieved chunks
            max_retries: Maximum re-generation attempts

        Returns:
            Generation result with verification info
        """
        if not self.generator:
            raise ValueError("Generator not provided to AnswerRefiner")

        # Extract requested period from question
        from src.retrieval.query_processor import FiscalPeriodExtractor
        extractor = FiscalPeriodExtractor()
        fiscal_info = extractor.extract(question)
        requested_period = fiscal_info.get("raw")

        # Initial generation
        result = self.generator.generate(question, chunks)

        # Verify answer
        verification = self.verifier.verify(
            answer=result.get("answer", ""),
            context=result.get("context_used", ""),
            chunks=chunks,
            requested_period=requested_period,
        )

        result["verification"] = verification

        # If critical issues and retries allowed, try again with enhanced prompt
        if not verification["is_valid"] and max_retries > 0:
            logger.warning(
                f"AnswerRefiner: Initial answer had issues: {verification['issues']}. "
                f"Retrying with enhanced prompt."
            )

            # Add verification feedback to prompt
            enhanced_question = self._enhance_question(
                question,
                verification["issues"]
            )

            retry_result = self.generator.generate(enhanced_question, chunks)
            retry_verification = self.verifier.verify(
                answer=retry_result.get("answer", ""),
                context=retry_result.get("context_used", ""),
                chunks=chunks,
                requested_period=requested_period,
            )

            # Use retry if it's better
            if retry_verification["confidence"] > verification["confidence"]:
                result = retry_result
                result["verification"] = retry_verification
                result["was_refined"] = True

        return result

    def _enhance_question(self, question: str, issues: List[str]) -> str:
        """Enhance question based on verification issues."""
        enhancements = []

        if any("citation" in issue.lower() for issue in issues):
            enhancements.append(
                "You MUST cite your sources using [Doc-N] format."
            )

        if any("period" in issue.lower() or "temporal" in issue.lower() for issue in issues):
            enhancements.append(
                "Verify that your answer uses data from the correct fiscal period."
            )

        if any("number" in issue.lower() or "hallucination" in issue.lower() for issue in issues):
            enhancements.append(
                "Only use numbers that appear verbatim in the documents."
            )

        if enhancements:
            return question + "\n\nIMPORTANT: " + " ".join(enhancements)
        return question


def extract_citation_refs(text: str) -> List[int]:
    """
    Extract citation references from text.

    Args:
        text: Text containing [Doc-N] citations

    Returns:
        List of document numbers cited
    """
    pattern = r'\[Doc-(\d+)\]'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


def validate_citations(
    answer: str,
    num_chunks: int,
) -> Tuple[bool, List[str]]:
    """
    Validate that all citations in answer reference valid documents.

    Args:
        answer: Generated answer text
        num_chunks: Number of chunks provided in context

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    citations = extract_citation_refs(answer)
    issues = []

    for doc_num in citations:
        if doc_num < 1 or doc_num > num_chunks:
            issues.append(
                f"Invalid citation [Doc-{doc_num}]: only {num_chunks} documents available"
            )

    return len(issues) == 0, issues
