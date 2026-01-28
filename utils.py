# src/genai/mutlagentic/utils.py

"""
Contains utility classes and functions for the agent graph.
- AgentState: The definition of the graph's state.
- should_continue: The conditional routing logic.
"""

import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import END

# ---------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------
class AgentState(TypedDict):
    """
    The state of the agent.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    # We add a new key to the state to hold the current date.
    current_date: str

# ---------------------------------------------------------------
# Graph Routing
# ---------------------------------------------------------------
def should_continue(state: AgentState):
    """
    This is the router. It decides whether to:
    1. Call a tool
    2. End the conversation
    """
    last_message = state['messages'][-1]
    # If the LLM's last response was a tool call,
    # we route to the 'action' node.
    if last_message.tool_calls:
        return "action"
    # Otherwise, the LLM has given a final answer,
    # so we end the graph.
    else:
        return END
