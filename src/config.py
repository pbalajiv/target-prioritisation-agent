"""
Configuration for the Target Prioritisation Agent.

Centralises all API endpoints and model settings in one place.
If an endpoint changes (OpenTargets ships a new API version, for instance),
you only need to update it here rather than hunting through the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM settings ---
# Using Google Gemini — free tier available at https://aistudio.google.com/
# Alternatives: set ANTHROPIC_API_KEY and use ChatAnthropic in agent.py
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

# --- OpenTargets Platform API ---
# Public GraphQL endpoint, no auth needed.
# Docs: https://platform-docs.opentargets.org/data-access/graphql-api
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# --- GTEx API ---
# Public REST endpoint for the GTEx Portal v2 API.
# Docs: https://gtexportal.org/api/v2/redoc
GTEX_API_BASE = "https://gtexportal.org/api/v2"
