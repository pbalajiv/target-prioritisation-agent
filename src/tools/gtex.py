"""
GTEx Portal API wrapper.

GTEx (Genotype-Tissue Expression project) provides gene expression data
across 54 human tissues from ~1000 post-mortem donors.  For target
prioritisation, tissue expression matters because:

  - A target expressed only in tumour-relevant tissue is more likely to
    have a tissue-specific function (good for efficacy, good for safety).
  - A target expressed ubiquitously raises safety concerns — inhibiting
    it could cause off-target effects in healthy organs.
  - Expression in immune cell-rich tissues (spleen, blood) versus
    structural tissues (muscle, skin) hints at whether the target's role
    is immunological versus structural.

We query the median TPM (transcripts per million) per tissue for a given
gene.  TPM normalises for gene length and sequencing depth, making it
comparable across tissues.
"""

import requests
from src.config import GTEX_API_BASE


def resolve_gene_to_gencode_id(gene_symbol: str) -> str | None:
    """
    Resolve a human-readable gene symbol (e.g., 'TYK2') to a GENCODE ID
    (e.g., 'ENSG00000105397.14') using the GTEx gene search endpoint.

    Why this is needed: The GTEx medianGeneExpression endpoint requires a
    GENCODE versioned Ensembl ID, not a gene symbol.  Without this lookup
    step, the API silently returns empty results — which is exactly the
    bug we hit in the first run.

    Args:
        gene_symbol: HGNC gene symbol, e.g. "TYK2"

    Returns:
        GENCODE ID string, or None if lookup fails
    """
    url = f"{GTEX_API_BASE}/reference/gene"
    params = {"geneId": gene_symbol, "datasetId": "gtex_v8"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # The API returns a list of matching genes
        genes = data.get("data", [])
        if genes:
            return genes[0].get("gencodeId")
        return None

    except Exception as e:
        print(f"Gene symbol lookup failed for {gene_symbol}: {e}")
        return None


def get_gene_expression(gene_symbol: str, top_n: int = 10) -> list[dict]:
    """
    Retrieve median tissue-level expression for a gene from GTEx.

    Returns the top N tissues by median TPM, which tells us where this
    gene is most actively transcribed.

    Args:
        gene_symbol: HGNC gene symbol, e.g. "FLT3", "LILRB4"
        top_n: Number of top-expressing tissues to return

    Returns:
        List of dicts with keys: tissue, median_tpm
        Sorted descending by expression level
    """
    # The GTEx v2 API endpoint for median gene expression across tissues.
    # Step 1: Resolve gene symbol to GENCODE ID — the API needs this format
    gencode_id = resolve_gene_to_gencode_id(gene_symbol)
    if not gencode_id:
        print(f"Could not resolve '{gene_symbol}' to a GENCODE ID")
        return []

    url = f"{GTEX_API_BASE}/expression/medianGeneExpression"
    params = {
        "gencodeId": gencode_id,
        "datasetId": "gtex_v8",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("data", [])
        if not records:
            return []

        # Sort by median TPM descending and take top N
        sorted_records = sorted(
            records,
            key=lambda r: r.get("median", 0),
            reverse=True,
        )

        results = []
        for rec in sorted_records[:top_n]:
            tissue_id = rec.get("tissueSiteDetailId", "unknown")
            # Make tissue names more readable:
            # "Brain_Cerebellum" -> "Brain - Cerebellum"
            tissue_label = tissue_id.replace("_", " - ", 1).replace("_", " ")
            results.append({
                "tissue": tissue_label,
                "median_tpm": round(rec.get("median", 0), 2),
            })

        return results

    except Exception as e:
        print(f"GTEx query failed for {gene_symbol}: {e}")
        return []


def query_gtex(gene_symbol: str) -> str:
    """
    Get tissue expression profile for a gene and return a formatted summary.

    This is the function the LangChain tool calls.  Like the OpenTargets
    wrapper, we return readable text rather than raw JSON.

    The summary helps the LLM reason about:
      - Where the gene is most expressed (tissue specificity)
      - Whether expression is broad or restricted (safety signal)
      - Whether the expression pattern matches the disease biology

    Args:
        gene_symbol: HGNC gene symbol

    Returns:
        Formatted string with tissue expression profile
    """
    expression_data = get_gene_expression(gene_symbol, top_n=10)

    if not expression_data:
        return (
            f"No expression data found for '{gene_symbol}' in GTEx. "
            f"The gene symbol may be incorrect, or the gene may not be "
            f"well-represented in the GTEx dataset."
        )

    lines = [
        f"Tissue expression profile for {gene_symbol} (GTEx v8):",
        f"Top {len(expression_data)} tissues by median TPM:\n",
    ]

    for i, entry in enumerate(expression_data, 1):
        # Add a rough qualitative label to help the LLM interpret
        tpm = entry["median_tpm"]
        if tpm > 100:
            level = "very high"
        elif tpm > 30:
            level = "high"
        elif tpm > 5:
            level = "moderate"
        elif tpm > 1:
            level = "low"
        else:
            level = "very low"

        lines.append(
            f"  {i}. {entry['tissue']}: {tpm} TPM ({level})"
        )

    # Add a brief interpretation hint for the LLM
    top_tissue = expression_data[0]["tissue"]
    lines.append(
        f"\nHighest expression is in {top_tissue}. "
        f"Consider whether this aligns with the disease tissue of interest."
    )

    return "\n".join(lines)
