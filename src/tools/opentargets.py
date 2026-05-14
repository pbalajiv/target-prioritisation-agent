"""
OpenTargets Platform API wrapper.

OpenTargets aggregates evidence linking genes/proteins to diseases from
GWAS, somatic mutations, known drugs, text mining, animal models, and more.
Each association gets an overall score (0-1) reflecting the combined weight
of all evidence sources.

We expose two operations the agent might need:
  1. search_disease  — fuzzy-match a disease name to an EFO ontology ID
  2. get_targets     — retrieve the top associated targets for a disease

Both return plain-text summaries (not raw JSON) because the LLM will reason
over the output directly.  Keeping the text concise avoids burning context
window tokens on fields the agent doesn't need.
"""

import requests
from typing import Optional
from src.config import OPENTARGETS_GRAPHQL_URL


def search_disease(disease_name: str, size: int = 5) -> list[dict]:
    """
    Search OpenTargets for a disease by name and return matching entries.

    Why this step exists: OpenTargets uses EFO ontology IDs internally
    (e.g., EFO_0000616 for ulcerative colitis).  The user types a disease
    name in plain English, so we need to resolve it to an ID first.

    Args:
        disease_name: Free-text disease name, e.g. "ulcerative colitis"
        size: Max number of matches to return

    Returns:
        List of dicts with keys: id, name, description
    """
    query = """
    query SearchDisease($queryString: String!, $size: Int!) {
      search(queryString: $queryString, entityNames: ["disease"], page: {size: $size, index: 0}) {
        hits {
          id
          name
          description
        }
      }
    }
    """
    variables = {"queryString": disease_name, "size": size}

    try:
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("search", {}).get("hits", [])

    except Exception as e:
        # We return an empty list rather than crashing so the agent can
        # gracefully tell the user the API is down and try again later.
        print(f"OpenTargets search failed: {e}")
        return []


def get_targets_for_disease(disease_id: str, top_n: int = 10) -> list[dict]:
    """
    Get the top gene targets associated with a disease from OpenTargets.

    This is the core query for target identification.  The association score
    reflects combined evidence from genetics (GWAS, gene burden), somatic
    mutations, drugs, pathways, literature, and animal models.

    A score above 0.7 generally means strong multi-source evidence.
    Scores below 0.3 are typically driven by a single weak source
    (often text mining alone).

    Args:
        disease_id: EFO ontology ID, e.g. "EFO_0000616"
        top_n: Number of top targets to return (default 10)

    Returns:
        List of dicts with keys: target_id, gene_symbol, target_name,
        overall_score, and per-datatype scores
    """
    query = """
    query TargetsForDisease($diseaseId: String!, $size: Int!) {
      disease(efoId: $diseaseId) {
        name
        associatedTargets(page: {size: $size, index: 0}) {
          rows {
            target {
              id
              approvedSymbol
              approvedName
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """
    variables = {"diseaseId": disease_id, "size": top_n}

    try:
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        disease_data = data.get("data", {}).get("disease")
        if not disease_data:
            return []

        rows = disease_data.get("associatedTargets", {}).get("rows", [])
        results = []
        for row in rows:
            target = row.get("target", {})

            # Flatten the datatype scores into a readable dict.
            # These correspond to OpenTargets evidence categories:
            # genetic_association, somatic_mutation, known_drug, etc.
            datatype_scores = {
                dt["id"]: round(dt["score"], 3)
                for dt in row.get("datatypeScores", [])
                if dt["score"] > 0
            }

            results.append({
                "target_id": target.get("id", ""),
                "gene_symbol": target.get("approvedSymbol", ""),
                "target_name": target.get("approvedName", ""),
                "overall_score": round(row.get("score", 0), 3),
                "evidence_scores": datatype_scores,
            })

        return results

    except Exception as e:
        print(f"OpenTargets target query failed: {e}")
        return []


def query_opentargets(disease_name: str, top_n: int = 10) -> str:
    """
    End-to-end convenience function: takes a disease name in plain English,
    resolves it to an EFO ID, and returns the top associated targets as a
    formatted text summary.

    This is the function the LangChain tool will actually call.  We return
    a human-readable string (not JSON) because the LLM needs to reason over
    it, and a clean summary is easier to reason over than nested JSON.

    Args:
        disease_name: Plain-English disease name
        top_n: How many targets to retrieve

    Returns:
        Formatted string summarising the top targets and their evidence
    """
    # Step 1: resolve disease name to EFO ID
    matches = search_disease(disease_name, size=3)
    if not matches:
        return f"No disease found matching '{disease_name}' in OpenTargets."

    # Take the top match — OpenTargets search is usually good at ranking
    best_match = matches[0]
    disease_id = best_match["id"]
    disease_label = best_match["name"]

    # Step 2: get associated targets
    targets = get_targets_for_disease(disease_id, top_n=top_n)
    if not targets:
        return f"Found disease '{disease_label}' ({disease_id}) but no associated targets."

    # Step 3: format into a readable summary
    lines = [
        f"Disease: {disease_label} ({disease_id})",
        f"Top {len(targets)} associated targets from OpenTargets:\n",
    ]
    for i, t in enumerate(targets, 1):
        evidence_parts = [f"{k}: {v}" for k, v in t["evidence_scores"].items()]
        evidence_str = ", ".join(evidence_parts) if evidence_parts else "no breakdown available"
        lines.append(
            f"  {i}. {t['gene_symbol']} ({t['target_name']})\n"
            f"     Overall score: {t['overall_score']}\n"
            f"     Evidence: {evidence_str}"
        )

    return "\n".join(lines)
