"""
Target Prioritisation Agent.

This module wires together the OpenTargets and GTEx tools with Claude
in a ReAct-style agent loop.  Here's what happens at runtime:

  1. The user provides a disease name (e.g., "acute myeloid leukemia").
  2. The LLM reads the system prompt, which tells it to act as a
     translational scientist and use the available tools.
  3. The LLM decides what to do first — typically it calls the
     OpenTargets tool to get disease-target associations.
  4. The tool result comes back as text.  The LLM reads it, reasons
     about what it learned, and decides whether it needs more info
     (e.g., checking tissue expression for top hits via GTEx).
  5. This loop continues until the LLM decides it has enough evidence
     to produce a final ranked list with mechanistic reasoning.

The key architectural choice: we use LangChain's tool-calling agent
rather than a manual ReAct prompt.  Tool-calling agents use the LLM's
native function-calling capability, which is more reliable than asking
the model to output structured "Action: ..." text and parsing it with
regex.  Claude's tool-calling is well-supported through langchain-anthropic.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from src.tools.opentargets import query_opentargets
from src.tools.gtex import query_gtex
from src.config import GOOGLE_API_KEY, MODEL_NAME


# --- Define LangChain tools ---
# The @tool decorator converts a plain Python function into something
# LangChain (and the LLM) can discover and call.  The docstring becomes
# the tool description — this is what the LLM reads to decide WHEN to
# use each tool.  Write it for the LLM, not for a developer.

@tool
def opentargets_tool(disease_name: str) -> str:
    """
    Query OpenTargets for the top gene targets associated with a disease.
    Input should be a disease name in plain English (e.g., 'ulcerative colitis',
    'acute myeloid leukemia', 'Crohn's disease').
    Returns a ranked list of targets with association scores and evidence
    type breakdown (genetic, somatic, drug, literature, etc.).
    Use this tool FIRST to identify candidate targets for a disease.
    """
    return query_opentargets(disease_name)


@tool
def gtex_tool(gene_symbol: str) -> str:
    """
    Query GTEx for the tissue expression profile of a gene.
    Input should be a standard HGNC gene symbol (e.g., 'FLT3', 'JAK2', 'LILRB4').
    Returns median TPM expression across the top tissues where this gene
    is most expressed.
    Use this tool AFTER identifying candidate targets to check whether
    their expression pattern is relevant to the disease tissue.
    """
    return query_gtex(gene_symbol)


# System prompt — this shapes how the agent reasons.
# Written to mirror how an actual translational scientist thinks through
# target prioritisation, so the output is scientifically meaningful
# rather than generic LLM boilerplate.
SYSTEM_PROMPT = """You are a translational scientist with expertise in target
identification and validation. Your task is to prioritise drug targets for
a given disease indication.

Your approach should follow this workflow:
1. First, use the OpenTargets tool to find targets genetically or otherwise
   associated with the disease.
2. For the most promising targets (top 3-5 by score), use the GTEx tool to
   check their tissue expression profile. Tissue specificity matters for
   both efficacy and safety.
3. Synthesise everything into a ranked list of target recommendations.

For each recommended target, provide:
- The gene symbol and full name
- Why the genetic/association evidence supports it (cite the OpenTargets score
  and which evidence types contribute most)
- Whether the tissue expression pattern fits the disease biology
- A brief mechanistic hypothesis for how this target might be involved
- Any safety concerns based on expression breadth

Be specific and scientific. Avoid vague statements. If a target has weak
evidence or doesn't express in relevant tissue, say so — honest assessment
is more valuable than enthusiasm.

At the end, provide a clear ranked summary with your top 3 recommendations
and a one-sentence rationale for each."""


def create_agent():
    """
    Instantiate the target prioritisation agent.

    Returns a runnable agent that accepts messages and produces a response,
    calling tools as needed along the way.
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,     # low temp for more deterministic scientific reasoning
        max_output_tokens=4096,
    )

    tools = [opentargets_tool, gtex_tool]

    # create_react_agent builds a graph that implements the ReAct loop:
    #   LLM decides -> calls tool -> reads result -> decides again -> ...
    # The 'prompt' injects our system prompt at the start of every
    # turn so the LLM always has its scientific framing.
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def run_agent(disease_name: str) -> str:
    """
    Run the target prioritisation agent for a disease and return the
    final response.

    This is the main entry point — call this from the Streamlit app or
    from a script.

    Args:
        disease_name: Plain-English disease name

    Returns:
        The agent's final synthesised response with ranked targets

    Example:
        >>> result = run_agent("ulcerative colitis")
        >>> print(result)
    """
    agent = create_agent()

    user_message = (
        f"Identify and prioritise drug targets for {disease_name}. "
        f"Query the relevant databases, check tissue expression for the "
        f"most promising candidates, and provide a ranked assessment "
        f"with mechanistic reasoning."
    )

    # The agent returns a dict with a "messages" key containing the full
    # conversation (system prompt, tool calls, tool results, final answer).
    # We want just the last message — the agent's synthesised response.
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]}
    )

    # Extract the final AI message
    final_message = result["messages"][-1]
    content = final_message.content

    # Different LLM providers return content in different formats:
    # - Anthropic (Claude): returns a plain string
    # - Google (Gemini): returns a list of content blocks like
    #   [{'type': 'text', 'text': '...'}]
    # We normalise to a plain string here so the caller doesn't need
    # to know which provider is behind the agent.
    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)

    return content
