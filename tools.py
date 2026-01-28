# src/genai/mutlagentic/tools.py

"""
Defines the tools available to the agent.
- document_tool: A "Smart" RAG tool that finds companies from the query.
- web_search_tool: Tavily for live web search.
- calculator_tool: Safe math evaluation.
"""

import os
import math
import requests
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

# --- Import from our custom modules ---
from ..llm import query_ollama_rag
from ..retrivel.pince_ret import get_all_namespaces

# ---------------------------------------------------------------
# Initialize Tools
# ---------------------------------------------------------------
load_dotenv()
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
# Ensure Tavily is initialized only if key exists to avoid immediate crashes
if TAVILY_KEY:
    tavily = TavilyClient(api_key=TAVILY_KEY)
else:
    print("Warning: TAVILY_API_KEY not found. Web search will fail.")
    tavily = None

# ---------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------

@tool
def calculator_tool(expression: str) -> str:
    """
    Useful for performing mathematical calculations.
    Use this for math problems, calculating ratios, margins, or percentage differences
    based on data retrieved from other tools.
    input: a mathematical expression string like '200 / 5' or '(4500 - 3200) / 4500'.
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        # Cleanup expression just in case LLM adds weird characters
        cleaned_expression = expression.replace(",", "").strip()
        result = eval(cleaned_expression, {"__builtins__": {}}, allowed)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Calculation Error: {e}. Please ensure the expression is valid math."


@tool
def web_search_tool(query: str) -> str:
    """
    Useful for finding current events, latest news, real-time market data,
    or general knowledge present on the live internet.
    Use this specifically when the user asks for "latest" or "today's" information.
    """
    if not tavily:
        return "Error: Web search tool is not configured (missing API key)."

    try:
        print(f"--- Running Web Search for: {query} ---")
        results = tavily.search(query=query, max_results=3)
        if not results.get("results"):
            return "No relevant web results found."
        
        # Format results clearly for the LLM
        snippets = []
        for i, r in enumerate(results["results"]):
            snippets.append(f"Result {i+1} (Source: {r['url']}):\n{r['content']}\n")
            
        return "Web Search Results:\n" + "\n".join(snippets)
    except Exception as e:
        return f"Web Search Error: {e}"


@tool
def document_tool(query: str, company_names: Optional[List[str]] = None) -> str:
    """
    Primary tool for retrieving detailed information from internal documents,
    such as historical company 10-K filings, annual reports, and specific risk factors.
    Use this for in-depth analysis of a company's past performance or stated risks.

    Args:
        query (str): The specific question to ask the documents.
        company_names (Optional[List[str]]): Optional list of company names to narrow search.
    """
    # ... (Rest of the function remains exactly the same as your original code)
    print(f"Document Tool called with query: '{query}'")
    if company_names:
        print(f"Explicit company namespaces provided: {company_names}")
    else:
        print("No explicit company names. Attempting to match from query...")

    try:
        all_available_namespaces = get_all_namespaces()
        if not all_available_namespaces:
            return "Error: Could not retrieve any namespaces from Pinecone."
    except Exception as e:
        return f"Error fetching namespaces: {e}"

    # --- Namespace Logic ---
    validated_namespaces = []
    invalid_names = []

    if company_names:
        # --- SCENARIO 1: EXPLICIT MODE (List is provided, e.g., from UI) ---
        for name in company_names:
            normalized_name = name.lower().strip().replace(" ", "_")
            found_match = False
            for ns in all_available_namespaces:
                if normalized_name in ns:
                    if ns not in validated_namespaces:
                        validated_namespaces.append(ns)
                    found_match = True
            if not found_match:
                invalid_names.append(name)
    else:
        # --- SCENARIO 2: IMPLICIT MODE (List is NOT provided) ---
        # We "smartly" guess from the query string.
        query_lower = query.lower()
        for ns in all_available_namespaces:
            # 'amazon_namespace' -> 'amazon'
            simple_ns_name = ns.split('_')[0].lower()
            if simple_ns_name in query_lower:
                if ns not in validated_namespaces:
                    validated_namespaces.append(ns)

    if not validated_namespaces:
        error_msg = f"Error: No valid company namespaces found for the query into internal documents."
        if invalid_names:
            error_msg += f" Invalid names: {', '.join(invalid_names)}."
        error_msg += f"\nAvailable internal namespaces are: {', '.join(all_available_namespaces)}"
        return error_msg
    # --- End of Namespace Logic ---

    # --- RAG Loop for each *matched* namespace ---
    all_summaries = [
        f"Info: Successfully matched query to internal namespaces: {', '.join(validated_namespaces)}"
    ]
    if invalid_names:
        all_summaries.append(f"Info: Skipped invalid names: {', '.join(invalid_names)}")

    for ns in validated_namespaces:
        # We pass the *original query* to RAG, not a generic one
        print(f"Querying internal namespace: {ns} for query: '{query}'")
        try:
            summary = query_ollama_rag(query, ns)
            all_summaries.append(f"--- Internal Document Summary for {ns.upper()} ---\n{summary}")
        except Exception as e:
            all_summaries.append(f"Error processing namespace {ns}: {e}")

    return "\n\n".join(all_summaries)