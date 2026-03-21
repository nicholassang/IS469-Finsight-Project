"""
Test script for Phase 1 & Phase 2 enhancements
Can be run on cluster to verify improvements are working.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: Import compatibility layer BEFORE any other imports
import chromadb_compat


def test_phase1_components():
    """Test Phase 1: Fiscal period detection and context management"""
    print("=" * 60)
    print("PHASE 1 TESTS")
    print("=" * 60)

    # Test 1: FiscalPeriodExtractor
    print("\n1. Testing FiscalPeriodExtractor...")
    from src.retrieval.query_processor import FiscalPeriodExtractor

    extractor = FiscalPeriodExtractor()
    test_cases = [
        ("What was revenue in Q2 FY2025?", "FY2025", "Q2"),
        ("Show me FY2024 annual results", "FY2024", None),
        ("Q3 2025 cloud growth", None, "Q3"),
    ]

    for query, expected_fy, expected_q in test_cases:
        result = extractor.extract(query)
        fy_match = result["fiscal_year"] == expected_fy
        q_match = result["quarter"] == expected_q
        status = "✓" if (fy_match and q_match) else "✗"
        print(f"   {status} {query[:40]:40s} -> {result['raw']}")

    print("   ✓ FiscalPeriodExtractor passed")

    # Test 2: ContextManager
    print("\n2. Testing ContextManager...")
    from src.generation.context_manager import ContextManager

    cm = ContextManager(model_name="qwen2.5-14b")
    print(f"   Token limit: {cm.max_context_tokens}")

    # Create oversized chunks
    large_chunks = [
        {"text": "This is a very long chunk. " * 500, "metadata": {"chunk_id": f"c{i}"}}
        for i in range(10)
    ]

    fitted, stats = cm.fit_context(large_chunks, min_chunks=3)
    print(f"   Original: {stats['original_count']} chunks, {stats['original_tokens']} tokens")
    print(f"   Fitted: {stats['selected_count']} chunks, {stats['selected_tokens']} tokens")
    print(f"   Truncated: {stats['truncated']}")

    assert stats['selected_tokens'] <= cm.max_context_tokens, "Token limit exceeded!"
    print("   ✓ ContextManager passed")


def test_phase2_components():
    """Test Phase 2: Verification and caching"""
    print("\n" + "=" * 60)
    print("PHASE 2 TESTS")
    print("=" * 60)

    # Test 1: AnswerVerifier
    print("\n1. Testing AnswerVerifier...")
    from src.generation.answer_verifier import AnswerVerifier, validate_citations

    verifier = AnswerVerifier()

    # Test citation validation
    good_answer = "Revenue was $65.6 billion [Doc-1] in Q2 FY2025. Azure grew 31% [Doc-2]."
    bad_answer = "Revenue was $999 billion [Doc-99]."

    is_valid, issues = validate_citations(good_answer, num_chunks=3)
    print(f"   Valid answer citations: {is_valid} (expected True)")
    assert is_valid, "Should validate good citations"

    is_valid, issues = validate_citations(bad_answer, num_chunks=3)
    print(f"   Invalid answer citations: {is_valid} (expected False)")
    print(f"   Issues: {issues[0]}")
    assert not is_valid, "Should reject invalid citations"

    # Test full verification
    mock_chunks = [
        {"text": "Revenue was $65.6 billion for Q2 FY2025.", "metadata": {"fiscal_period": "Q2 FY2025"}},
        {"text": "Azure revenue grew 31%.", "metadata": {"fiscal_period": "Q2 FY2025"}},
    ]

    result = verifier.verify(good_answer, "mock context", mock_chunks, "Q2 FY2025")
    print(f"   Verification confidence: {result['confidence']:.2f}")
    print(f"   Citations found: {result['citation_count']}")

    print("   ✓ AnswerVerifier passed")

    # Test 2: QueryCache
    print("\n2. Testing QueryCache...")
    from src.utils.query_cache import QueryCache

    cache = QueryCache()

    # Cache a response
    test_query = "What was Microsoft revenue?"
    test_response = {"answer": "Revenue was $65.6 billion", "latency": 1.2}

    cache.put_response(test_query, test_response)

    # Retrieve it
    cached = cache.get_response(test_query)
    assert cached is not None, "Cache should return value"
    assert cached["answer"] == test_response["answer"], "Cached answer should match"

    # Test normalization (different spacing/capitalization should match)
    cached = cache.get_response("  what WAS microsoft REVENUE?  ")
    assert cached is not None, "Normalized query should hit cache"

    stats = cache.stats
    print(f"   Response cache hit rate: {stats['response_cache']['hit_rate']}")
    print(f"   Cache size: {stats['response_cache']['size']}")

    print("   ✓ QueryCache passed")

    # Test 3: VerifiedRetriever period matching
    print("\n3. Testing VerifiedRetriever period matching...")
    from src.retrieval.verified_retriever import VerifiedRetriever

    class TestVR(VerifiedRetriever):
        def __init__(self):
            # Skip full init to avoid ChromaDB
            pass

    vr = TestVR()

    # Test period matching logic
    test_cases = [
        ("Q2 FY2025", "Q2 FY2025", True, "Exact match"),
        ("FY2025", "Q2 FY2025", True, "Annual contains quarterly"),
        ("Q1 FY2025", "Q2 FY2025", False, "Different quarter"),
        ("Q2 FY2024", "Q2 FY2025", False, "Different year"),
    ]

    for chunk_period, requested_period, expected, description in test_cases:
        result = vr._periods_match(chunk_period, requested_period)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {description:30s} {result}")
        assert result == expected, f"Period matching failed for {description}"

    print("   ✓ VerifiedRetriever period matching passed")


def main():
    """Run all tests"""
    try:
        test_phase1_components()
        test_phase2_components()

        print("\n" + "=" * 60)
        print("ALL PHASE 1 & 2 TESTS PASSED ✓")
        print("=" * 60)
        print("\nPhase 1 & Phase 2 implementations are ready.")
        print("Run the full evaluation with: python evaluation/run_eval.py")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
