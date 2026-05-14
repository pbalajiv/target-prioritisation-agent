"""
Integration tests — run these manually with real API keys.

These hit live APIs (OpenTargets, GTEx, Anthropic) and take 30-60 seconds.
Don't include them in CI/CD unless you have a budget for API calls.

Run with:
    pytest tests/test_integration.py -v -s

The -s flag shows print output so you can see the agent's reasoning.
"""

import os
import pytest
from src.tools.opentargets import search_disease, get_targets_for_disease, query_opentargets
from src.tools.gtex import get_gene_expression, query_gtex


# Skip all tests in this file if no API key is set
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping integration tests",
)


class TestOpenTargetsLive:
    """Tests against the real OpenTargets API."""

    def test_search_known_disease(self):
        """Ulcerative colitis is a well-known IBD — should always return hits."""
        results = search_disease("ulcerative colitis")
        assert len(results) > 0
        # The top hit should be the correct disease
        assert "colitis" in results[0]["name"].lower()

    def test_get_targets_returns_scored_list(self):
        """EFO_0000616 is ulcerative colitis — should have many targets."""
        targets = get_targets_for_disease("EFO_0000616", top_n=5)
        assert len(targets) > 0
        # Scores should be between 0 and 1
        for t in targets:
            assert 0 < t["overall_score"] <= 1
            assert t["gene_symbol"]  # should not be empty

    def test_end_to_end_query(self):
        """Full query_opentargets should return a formatted string."""
        result = query_opentargets("Crohn's disease", top_n=5)
        print("\n--- OpenTargets output ---")
        print(result)
        assert "Crohn" in result
        assert "Overall score" in result


class TestGTExLive:
    """Tests against the real GTEx API."""

    def test_known_gene_has_expression(self):
        """TP53 is expressed in virtually all tissues."""
        results = get_gene_expression("TP53", top_n=5)
        assert len(results) > 0
        # At least some tissue should have nonzero TPM
        assert any(r["median_tpm"] > 0 for r in results)

    def test_end_to_end_query(self):
        """Full query_gtex should return a formatted profile."""
        result = query_gtex("JAK2")
        print("\n--- GTEx output ---")
        print(result)
        assert "JAK2" in result


class TestFullAgentLive:
    """
    Full agent integration test — this calls the LLM and both APIs.
    Expect ~60 seconds runtime and a few cents in API cost.
    """

    def test_agent_produces_ranked_output(self):
        from src.agent import run_agent

        result = run_agent("acute myeloid leukemia")
        print("\n--- Agent output ---")
        print(result)

        # The agent should produce a non-empty response
        assert len(result) > 200
        # It should mention at least one gene
        # (AML has well-known targets like FLT3, IDH1/2, NPM1)
        result_upper = result.upper()
        known_aml_targets = ["FLT3", "IDH1", "IDH2", "NPM1", "KIT", "TP53"]
        found = [g for g in known_aml_targets if g in result_upper]
        assert len(found) > 0, f"Expected at least one known AML target, found none in output"
