# 🧬 Target Prioritisation Agent

An AI agent that automates the early stages of drug target identification. Given a disease indication, it queries genetic association databases and tissue expression atlases, then synthesises a ranked list of target hypotheses with mechanistic reasoning.

## Why this exists

Target identification in pharma typically involves a scientist manually pulling data from OpenTargets, checking tissue expression in GTEx, reading relevant literature, and mentally integrating all of it into a shortlist. This project automates that workflow using an LLM-based agent that can decide which databases to query, interpret the results, and produce a reasoned assessment — the same loop a translational scientist runs, but faster and more systematic.

The key distinction from a simple script is that the agent **reasons about what to do next** based on what it finds. If a top-scoring target from OpenTargets turns out to have ubiquitous expression (a safety concern), the agent notes that and adjusts its ranking. The sequence of queries isn't hardcoded and the system adapts.

## Architecture

```
User input (disease name)
        │
        ▼
   ┌─────────┐
   │  Claude  │ ◄── ReAct loop: Thought → Action → Observation → ...
   │  (LLM)  │
   └────┬─────┘
        │ decides which tool to call
   ┌────┴─────────────────┐
   │                      │
   ▼                      ▼
┌──────────┐      ┌──────────┐
│OpenTargets│     │   GTEx    │
│ GraphQL   │     │ REST API  │
│           │     │           │
│ genetic   │     │ tissue    │
│ evidence  │     │ expression│
└──────────┘      └──────────┘
        │                 │
        └────────┬────────┘
                 ▼
    Ranked targets with mechanistic
    rationale and evidence synthesis
```

## What the agent does

1. Takes a disease name as input (e.g., "acute myeloid leukemia")
2. Queries **OpenTargets** to find genetically associated targets with evidence scores
3. For the most promising targets, queries **GTEx** to check tissue-specific expression
4. Synthesises findings into a ranked list with:
   - Genetic evidence summary per target
   - Tissue expression context (is it expressed where the disease manifests?)
   - Mechanistic hypothesis
   - Safety considerations from expression breadth

## Tech stack

| Component | Purpose |
|-----------|---------|
| Claude (Anthropic) | Reasoning engine — decides which tools to call and synthesises results |
| LangChain + LangGraph | Agent orchestration framework (ReAct loop) |
| OpenTargets GraphQL API | Disease-target genetic associations |
| GTEx REST API | Tissue-level gene expression (median TPM) |
| Streamlit | Web interface for demo |

## Setup

```bash
# Clone and enter directory
git clone https://github.com/YOUR_USERNAME/target-prioritisation-agent.git
cd target-prioritisation-agent

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Usage

**Web interface:**
```bash
streamlit run app.py
```

**Command line:**
```python
from src.agent import run_agent

result = run_agent("ulcerative colitis")
print(result)
```

## Running tests

```bash
# Unit tests (no API key needed, all HTTP calls mocked)
pytest tests/test_opentargets.py tests/test_gtex.py tests/test_agent.py -v

# Integration tests (needs API key, hits live APIs)
pytest tests/test_integration.py -v -s
```

## Example output

Input: `acute myeloid leukemia`

The agent typically identifies targets like FLT3, IDH1/2, NPM1, and KIT, noting that FLT3 has strong genetic evidence and high bone marrow expression (relevant tissue), while also flagging broadly expressed targets as potential safety concerns.

## Limitations

- The agent's reasoning quality depends on the LLM, and so it can occasionally miss nuances a domain expert would catch.
- OpenTargets scores aggregate heterogeneous evidence; the agent interprets these but doesn't critically evaluate the underlying study quality.
- GTEx expression data comes from healthy donors and not diseased tissue. Therefore expression may differ in disease states.
- This is a hypothesis generation tool, not a replacement for experimental validation.

## Scientific context

This project demonstrates **agentic AI for hypothesis generation** in translational science. The agent pattern (LLM + tool use + iterative reasoning) is applicable to many pharma workflows where a scientist currently integrates evidence from multiple databases manually. The approach is transparent and the agent shows its reasoning chain, making it auditable.

This is a work in progress and will be updated regularly as more advances are made.
