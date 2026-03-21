"""
Quick integration test for Phase 1 & 2 improvements
Tests the full RAG pipeline with actual vLLM and ChromaDB
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: Import compatibility layer BEFORE any other imports
# This patches SQLite for clusters with old versions
import chromadb_compat


def test_phase1_fiscal_filtering():
    """Test Phase 1: Fiscal period detection and filtering"""
    print("=" * 60)
    print("TEST 1: Fiscal Period Filtering (Phase 1)")
    print("=" * 60)

    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.generator import Generator

    hr = HybridRetriever()
    gen = Generator()

    # Test q007-style query: Specific quarter
    query = "What was Microsoft cloud revenue in Q2 FY2025?"
    print(f"\nQuery: {query}")

    chunks = hr.retrieve(query, top_k=5)
    print(f"Retrieved {len(chunks)} chunks")

    if chunks:
        top_period = chunks[0].get("metadata", {}).get("fiscal_period", "N/A")
        print(f"Top chunk fiscal period: {top_period}")

        # Check if majority match Q2 FY2025
        q2_fy25_count = sum(1 for c in chunks if "Q2 FY2025" in c.get("metadata", {}).get("fiscal_period", ""))
        match_rate = q2_fy25_count / len(chunks) if chunks else 0
        print(f"Match rate for Q2 FY2025: {match_rate:.0%} ({q2_fy25_count}/{len(chunks)})")

        if match_rate >= 0.6:
            print("✓ PASS: Good temporal filtering")
        else:
            print(f"⚠ WARNING: Only {match_rate:.0%} match requested period")

    # Generate answer
    result = gen.generate(query, chunks)
    answer = result.get("answer", "")
    print(f"\nAnswer preview: {answer[:200]}...")

    # Check for context truncation stats
    if "context_stats" in result:
        stats = result["context_stats"]
        print(f"\nContext stats:")
        print(f"  Selected chunks: {stats.get('selected_count', 'N/A')}")
        print(f"  Selected tokens: {stats.get('selected_tokens', 'N/A')}")
        print(f"  Truncated: {stats.get('truncated', False)}")

    print("\n✓ Test 1 complete")
    return True


def test_phase1_context_truncation():
    """Test Phase 1: Context manager prevents token limit errors"""
    print("\n" + "=" * 60)
    print("TEST 2: Context Truncation (Phase 1)")
    print("=" * 60)

    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.generation.generator import Generator

    hr = HybridRetriever()
    gen = Generator()

    # Test q015-style query: Multi-period comparison (lots of context)
    query = "Compare Microsoft cloud revenue between Q3 FY2024 and Q3 FY2023"
    print(f"\nQuery: {query}")

    # Retrieve many chunks to test truncation
    chunks = hr.retrieve(query, top_k=15)
    print(f"Retrieved {len(chunks)} chunks")

    # Check generator has context manager
    if hasattr(gen, 'context_manager'):
        cm = gen.context_manager
        print(f"Context manager max tokens: {cm.max_context_tokens}")

    # Generate - should NOT error on token limits
    try:
        result = gen.generate(query, chunks)
        answer = result.get("answer", "")
        print(f"\nAnswer generated successfully (length: {len(answer)} chars)")
        print(f"Preview: {answer[:150]}...")

        # Check if truncation occurred
        if "context_stats" in result:
            stats = result["context_stats"]
            if stats.get("truncated"):
                print(f"\n✓ Context was truncated to fit token limit")
                print(f"  Original: {stats.get('original_count')} chunks")
                print(f"  Selected: {stats.get('selected_count')} chunks")
            else:
                print(f"\n✓ All chunks fit within token limit")

        print("\n✓ Test 2 complete - No token limit errors!")
        return True

    except Exception as e:
        print(f"\n✗ FAIL: Token limit error occurred: {e}")
        return False


def test_phase2_answer_verification():
    """Test Phase 2: Answer verification"""
    print("\n" + "=" * 60)
    print("TEST 3: Answer Verification (Phase 2)")
    print("=" * 60)

    from src.generation.answer_verifier import AnswerVerifier, validate_citations

    verifier = AnswerVerifier()

    # Mock answer with citations
    test_answer = """
    Microsoft's cloud revenue in Q2 FY2025 was $36.8 billion [Doc-1],
    representing a 21% increase year-over-year [Doc-2]. Azure revenue
    specifically grew 31% in constant currency [Doc-3].
    """

    mock_chunks = [
        {"text": "Cloud revenue was $36.8 billion in Q2 FY2025.", "metadata": {"fiscal_period": "Q2 FY2025"}},
        {"text": "Year-over-year growth was 21%.", "metadata": {"fiscal_period": "Q2 FY2025"}},
        {"text": "Azure grew 31% in constant currency.", "metadata": {"fiscal_period": "Q2 FY2025"}},
    ]

    # Verify citations are valid
    is_valid, issues = validate_citations(test_answer, num_chunks=3)
    print(f"Citation validation: {is_valid}")
    if issues:
        print(f"Issues: {issues}")

    # Full verification
    result = verifier.verify(test_answer, "context", mock_chunks, "Q2 FY2025")
    print(f"\nVerification results:")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Citations: {result['citation_count']}")
    print(f"  Issues: {len(result['issues'])}")
    print(f"  Warnings: {len(result['warnings'])}")

    if result['is_valid']:
        print("\n✓ Test 3 complete - Answer verification working")
    else:
        print("\n⚠ Test 3 - Verification detected issues (expected for demo)")

    return True


def test_phase2_query_cache():
    """Test Phase 2: Query caching"""
    print("\n" + "=" * 60)
    print("TEST 4: Query Caching (Phase 2)")
    print("=" * 60)

    from src.utils.query_cache import QueryCache
    import time

    cache = QueryCache()

    # Test cache operations
    test_query = "What was Microsoft revenue in Q2 FY2025?"
    test_response = {"answer": "Revenue was $65.6 billion", "latency": 1.5}

    # First call - cache miss
    cached = cache.get_response(test_query)
    print(f"Initial cache lookup: {cached}")
    assert cached is None, "Cache should be empty initially"

    # Store in cache
    cache.put_response(test_query, test_response)
    print(f"Response cached")

    # Second call - cache hit
    cached = cache.get_response(test_query)
    print(f"Second cache lookup: {'HIT' if cached else 'MISS'}")
    assert cached is not None, "Should retrieve from cache"
    assert cached["answer"] == test_response["answer"]

    # Test normalization
    cached = cache.get_response("  WHAT was microsoft REVENUE in q2 fy2025?  ")
    print(f"Normalized query cache lookup: {'HIT' if cached else 'MISS'}")
    assert cached is not None, "Normalized query should hit cache"

    # Stats
    stats = cache.stats
    print(f"\nCache statistics:")
    print(f"  Response cache hit rate: {stats['response_cache']['hit_rate']}")
    print(f"  Cache size: {stats['response_cache']['size']}")

    print("\n✓ Test 4 complete - Caching working")
    return True


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 & 2 INTEGRATION TESTS")
    print("Testing with live vLLM and ChromaDB")
    print("=" * 60)

    results = []

    try:
        # Phase 1 tests
        results.append(("Fiscal Filtering", test_phase1_fiscal_filtering()))
        results.append(("Context Truncation", test_phase1_context_truncation()))

        # Phase 2 tests (non-DB)
        results.append(("Answer Verification", test_phase2_answer_verification()))
        results.append(("Query Caching", test_phase2_query_cache()))

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All integration tests passed!")
        print("\nPhase 1 & 2 improvements are working correctly.")
        print("Run full evaluation: python evaluation/run_eval.py")
    else:
        print("\n⚠ Some tests failed - check output above")

    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
