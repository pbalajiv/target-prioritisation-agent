"""
Tests for the agent module.

These tests verify the wiring — that tools are defined correctly, the
agent can be created, and the system prompt is in place.  We don't test
the full ReAct loop here (that would require a live LLM call and real
API keys), but we verify everything is assembled correctly.

For a full integration test, see test_integration.py (run manually
with a real API key).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agent import opentargets_tool, gtex_tool, SYSTEM_PROMPT, create_agent


# --- Tool definition tests ---

def test_opentargets_tool_has_correct_name():
    """LangChain tools expose a .name attribute the LLM sees."""
    assert opentargets_tool.name == "opentargets_tool"


def test_gtex_tool_has_correct_name():
    assert gtex_tool.name == "gtex_tool"


def test_opentargets_tool_has_description():
    """The description is what the LLM reads to decide when to call the tool.
    If it's empty, the LLM won't know what the tool does."""
    assert len(opentargets_tool.description) > 50


def test_gtex_tool_has_description():
    assert len(gtex_tool.description) > 50


def test_opentargets_tool_accepts_disease_name():
    """Verify the tool's input schema expects a disease_name string."""
    schema = opentargets_tool.args_schema.model_json_schema()
    assert "disease_name" in schema.get("properties", {})


def test_gtex_tool_accepts_gene_symbol():
    """Verify the tool's input schema expects a gene_symbol string."""
    schema = gtex_tool.args_schema.model_json_schema()
    assert "gene_symbol" in schema.get("properties", {})


# --- System prompt tests ---

def test_system_prompt_mentions_opentargets():
    """The prompt should guide the agent to use OpenTargets first."""
    assert "OpenTargets" in SYSTEM_PROMPT


def test_system_prompt_mentions_gtex():
    """The prompt should tell the agent about GTEx as a follow-up tool."""
    assert "GTEx" in SYSTEM_PROMPT


def test_system_prompt_mentions_ranking():
    """The agent should be instructed to produce a ranked output."""
    assert "ranked" in SYSTEM_PROMPT.lower()


# --- Agent creation tests ---

@patch("src.agent.GOOGLE_API_KEY", "fake-google-key-for-testing")
def test_create_agent_returns_runnable():
    """Verify the agent object is created and has an invoke method.
    This doesn't call the LLM — just checks the wiring is correct."""
    agent = create_agent()
    assert hasattr(agent, "invoke"), "Agent should be invocable"


# --- Tool invocation tests (mocked) ---

@patch("src.agent.query_opentargets")
def test_opentargets_tool_calls_wrapper(mock_query):
    """Verify the LangChain tool delegates to our wrapper function."""
    mock_query.return_value = "Disease: test\nTarget: ABC1"

    result = opentargets_tool.invoke({"disease_name": "test disease"})

    mock_query.assert_called_once_with("test disease")
    assert "ABC1" in result


@patch("src.agent.query_gtex")
def test_gtex_tool_calls_wrapper(mock_query):
    """Verify the LangChain tool delegates to our wrapper function."""
    mock_query.return_value = "FLT3 expression profile..."

    result = gtex_tool.invoke({"gene_symbol": "FLT3"})

    mock_query.assert_called_once_with("FLT3")
    assert "FLT3" in result
