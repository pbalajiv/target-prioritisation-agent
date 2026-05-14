"""
Tests for the OpenTargets API wrapper.

We mock all HTTP calls so these tests run fast and offline.
Each test verifies one behaviour:
  - happy path (API returns good data)
  - empty results (no matches)
  - API failure (network error, timeout)
  - end-to-end formatting (the string the agent actually sees)
"""

import pytest
from unittest.mock import patch, MagicMock
from src.tools.opentargets import (
    search_disease,
    get_targets_for_disease,
    query_opentargets,
)


# --- Fixtures: reusable fake API responses ---

def _mock_search_response():
    """Simulates the OpenTargets search endpoint returning a match."""
    return {
        "data": {
            "search": {
                "hits": [
                    {
                        "id": "EFO_0000616",
                        "name": "ulcerative colitis",
                        "description": "A chronic inflammatory bowel disease...",
                    }
                ]
            }
        }
    }


def _mock_targets_response():
    """Simulates the OpenTargets target associations endpoint."""
    return {
        "data": {
            "disease": {
                "name": "ulcerative colitis",
                "associatedTargets": {
                    "rows": [
                        {
                            "target": {
                                "id": "ENSG00000163599",
                                "approvedSymbol": "IL23R",
                                "approvedName": "interleukin 23 receptor",
                            },
                            "score": 0.85,
                            "datatypeScores": [
                                {"id": "genetic_association", "score": 0.92},
                                {"id": "known_drug", "score": 0.7},
                                {"id": "literature", "score": 0.45},
                            ],
                        },
                        {
                            "target": {
                                "id": "ENSG00000198001",
                                "approvedSymbol": "JAK2",
                                "approvedName": "Janus kinase 2",
                            },
                            "score": 0.72,
                            "datatypeScores": [
                                {"id": "genetic_association", "score": 0.6},
                                {"id": "known_drug", "score": 0.8},
                            ],
                        },
                    ]
                },
            }
        }
    }


# --- Tests for search_disease ---

@patch("src.tools.opentargets.requests.post")
def test_search_disease_returns_matches(mock_post):
    """Verify we correctly parse disease search results."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_search_response()
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    results = search_disease("ulcerative colitis")

    assert len(results) == 1
    assert results[0]["id"] == "EFO_0000616"
    assert results[0]["name"] == "ulcerative colitis"


@patch("src.tools.opentargets.requests.post")
def test_search_disease_empty_results(mock_post):
    """Verify we handle the case where no disease matches are found."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": {"search": {"hits": []}}}
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    results = search_disease("madeup disease xyz")
    assert results == []


@patch("src.tools.opentargets.requests.post")
def test_search_disease_api_failure(mock_post):
    """Verify we return empty list (not crash) when the API is down."""
    mock_post.side_effect = Exception("Connection refused")

    results = search_disease("ulcerative colitis")
    assert results == []


# --- Tests for get_targets_for_disease ---

@patch("src.tools.opentargets.requests.post")
def test_get_targets_parses_correctly(mock_post):
    """Verify we extract gene symbols, scores, and evidence breakdowns."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_targets_response()
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    targets = get_targets_for_disease("EFO_0000616")

    assert len(targets) == 2
    assert targets[0]["gene_symbol"] == "IL23R"
    assert targets[0]["overall_score"] == 0.85
    assert "genetic_association" in targets[0]["evidence_scores"]
    assert targets[1]["gene_symbol"] == "JAK2"


@patch("src.tools.opentargets.requests.post")
def test_get_targets_filters_zero_scores(mock_post):
    """Evidence types with score 0 should be excluded from the output."""
    response_data = _mock_targets_response()
    # Add a zero-score evidence type
    rows = response_data["data"]["disease"]["associatedTargets"]["rows"]
    rows[0]["datatypeScores"].append({"id": "animal_model", "score": 0.0})

    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    targets = get_targets_for_disease("EFO_0000616")
    assert "animal_model" not in targets[0]["evidence_scores"]


# --- Tests for query_opentargets (end-to-end formatting) ---

@patch("src.tools.opentargets.get_targets_for_disease")
@patch("src.tools.opentargets.search_disease")
def test_query_opentargets_formats_output(mock_search, mock_targets):
    """Verify the final text output the agent sees is well-structured."""
    mock_search.return_value = [
        {"id": "EFO_0000616", "name": "ulcerative colitis", "description": "..."}
    ]
    mock_targets.return_value = [
        {
            "target_id": "ENSG00000163599",
            "gene_symbol": "IL23R",
            "target_name": "interleukin 23 receptor",
            "overall_score": 0.85,
            "evidence_scores": {"genetic_association": 0.92, "known_drug": 0.7},
        }
    ]

    result = query_opentargets("ulcerative colitis")

    assert "ulcerative colitis" in result
    assert "IL23R" in result
    assert "0.85" in result
    assert "genetic_association" in result


@patch("src.tools.opentargets.search_disease")
def test_query_opentargets_no_disease_found(mock_search):
    """When the disease name doesn't match anything, we get a clear message."""
    mock_search.return_value = []

    result = query_opentargets("nonexistent disease")
    assert "No disease found" in result
