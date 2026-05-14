"""
Tests for the GTEx API wrapper.

Same pattern as the OpenTargets tests: mock HTTP calls, test parsing,
test error handling, test the formatted output the agent receives.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.tools.gtex import resolve_gene_to_gencode_id, get_gene_expression, query_gtex


def _mock_gene_lookup_response():
    """Simulates the GTEx gene reference endpoint."""
    return {
        "data": [
            {"gencodeId": "ENSG00000105397.14", "geneSymbol": "TYK2"}
        ]
    }


def _mock_gtex_response():
    """Simulates GTEx median expression endpoint for FLT3."""
    return {
        "data": [
            {"tissueSiteDetailId": "Whole_Blood", "median": 45.2},
            {"tissueSiteDetailId": "Spleen", "median": 38.7},
            {"tissueSiteDetailId": "Bone_Marrow", "median": 120.5},
            {"tissueSiteDetailId": "Liver", "median": 2.1},
            {"tissueSiteDetailId": "Brain_Cerebellum", "median": 0.3},
        ]
    }


# --- Tests for resolve_gene_to_gencode_id ---

@patch("src.tools.gtex.requests.get")
def test_resolve_gene_returns_gencode_id(mock_get):
    """Verify we correctly extract the GENCODE ID from the lookup."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gene_lookup_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = resolve_gene_to_gencode_id("TYK2")
    assert result == "ENSG00000105397.14"


@patch("src.tools.gtex.requests.get")
def test_resolve_gene_returns_none_for_unknown(mock_get):
    """Unknown gene symbols should return None, not crash."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = resolve_gene_to_gencode_id("FAKEGENE")
    assert result is None


# --- Tests for get_gene_expression ---

@patch("src.tools.gtex.resolve_gene_to_gencode_id")
@patch("src.tools.gtex.requests.get")
def test_get_gene_expression_sorts_by_tpm(mock_get, mock_resolve):
    """Results should come back sorted highest-to-lowest by TPM."""
    mock_resolve.return_value = "ENSG00000105397.14"
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gtex_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    results = get_gene_expression("FLT3", top_n=5)

    assert len(results) == 5
    # Bone marrow should be first (highest TPM)
    assert results[0]["tissue"] == "Bone - Marrow"
    assert results[0]["median_tpm"] == 120.5
    # TPM values should be in descending order
    tpm_values = [r["median_tpm"] for r in results]
    assert tpm_values == sorted(tpm_values, reverse=True)


@patch("src.tools.gtex.resolve_gene_to_gencode_id")
@patch("src.tools.gtex.requests.get")
def test_get_gene_expression_respects_top_n(mock_get, mock_resolve):
    """We should only return the requested number of tissues."""
    mock_resolve.return_value = "ENSG00000105397.14"
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gtex_response()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    results = get_gene_expression("FLT3", top_n=3)
    assert len(results) == 3


@patch("src.tools.gtex.resolve_gene_to_gencode_id")
def test_get_gene_expression_empty_when_resolve_fails(mock_resolve):
    """If gene symbol can't be resolved, return empty list."""
    mock_resolve.return_value = None

    results = get_gene_expression("FAKEGENE")
    assert results == []


@patch("src.tools.gtex.resolve_gene_to_gencode_id")
@patch("src.tools.gtex.requests.get")
def test_get_gene_expression_api_failure(mock_get, mock_resolve):
    """Network failures should return empty list, not crash."""
    mock_resolve.return_value = "ENSG00000105397.14"
    mock_get.side_effect = Exception("Timeout")

    results = get_gene_expression("FLT3")
    assert results == []


# --- Tests for query_gtex (formatted output) ---

@patch("src.tools.gtex.get_gene_expression")
def test_query_gtex_formats_output(mock_expr):
    """Verify the text summary includes gene name, tissues, and TPM values."""
    mock_expr.return_value = [
        {"tissue": "Bone - Marrow", "median_tpm": 120.5},
        {"tissue": "Whole - Blood", "median_tpm": 45.2},
        {"tissue": "Spleen", "median_tpm": 38.7},
    ]

    result = query_gtex("FLT3")

    assert "FLT3" in result
    assert "Bone - Marrow" in result
    assert "120.5" in result
    assert "very high" in result      # 120.5 TPM should be labeled "very high"
    assert "Highest expression" in result


@patch("src.tools.gtex.get_gene_expression")
def test_query_gtex_qualitative_labels(mock_expr):
    """Check that TPM values get the right qualitative labels."""
    mock_expr.return_value = [
        {"tissue": "TissueA", "median_tpm": 150.0},   # very high
        {"tissue": "TissueB", "median_tpm": 50.0},     # high
        {"tissue": "TissueC", "median_tpm": 10.0},     # moderate
        {"tissue": "TissueD", "median_tpm": 2.0},      # low
        {"tissue": "TissueE", "median_tpm": 0.5},      # very low
    ]

    result = query_gtex("TESTGENE")

    assert "very high" in result
    assert "high" in result
    assert "moderate" in result
    assert "low" in result
    assert "very low" in result


@patch("src.tools.gtex.get_gene_expression")
def test_query_gtex_no_data(mock_expr):
    """Missing gene should produce a helpful message, not an error."""
    mock_expr.return_value = []

    result = query_gtex("NONEXISTENT")
    assert "No expression data found" in result
    assert "NONEXISTENT" in result
