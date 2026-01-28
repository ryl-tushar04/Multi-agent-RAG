# src/genai/mutlagentic/graph.py

"""
Assembles and compiles the agent graph.
This is the simple "agent" -> "action" -> "agent" loop.
"""

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- Import from our custom modules ---
from ..llm import get_agent_brain  # <-- Only imports the "brain"
# --- 🚀 IMPORT THE CORRECTED TOOLS ---
from .tools import document_tool, web_search_tool, calculator_tool
from .prompt_hub import get_agent_prompt  # <-- Only imports the one prompt
from .utils import AgentState, should_continue

# ---------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------
def call_model(state: AgentState):
    """Calls the "Brain" LLM (Granite) to decide on logic AND formatting."""
    messages = state['messages']
    current_date = state['current_date']
    response = agent_llm.invoke({
        "messages": messages,
        "current_date": current_date
    })
    return {"messages": [response]}


def call_tools(state: AgentState):
    """Executes tools called by the "Brain"."""
    last_message = state['messages'][-1]
    tool_outputs = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        print(f"--- Calling Tool: {tool_name}({tool_args}) ---")

        if tool_name == "document_tool":
            selected_tool = document_tool
        elif tool_name == "web_search_tool":
            selected_tool = web_search_tool
        elif tool_name == "calculator_tool":
            selected_tool = calculator_tool
        else:
            tool_outputs.append(
                ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call['id'])
            )
            continue

        try:
            output = selected_tool.invoke(tool_args)
            tool_outputs.append(
                ToolMessage(content=str(output), tool_call_id=tool_call['id'])
            )
        except Exception as e:
            tool_outputs.append(
                ToolMessage(content=f"Error running tool: {e}", tool_call_id=tool_call['id'])
            )

    return {"messages": tool_outputs}


# ---------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------
load_dotenv()

# 1. Initialize tools and LLM
# --- 🚀 THE TOOLS LIST IS NOW CORRECT ---
tools = [document_tool, web_search_tool, calculator_tool]

# The "Brain" (Granite) for all logic and formatting
llm_brain = get_agent_brain()
llm_brain_with_tools = llm_brain.bind_tools(tools)
agent_prompt = get_agent_prompt()
agent_llm = agent_prompt | llm_brain_with_tools

# 2. Define the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", call_tools)
graph.set_entry_point("agent")

# 3. Add edges
graph.add_conditional_edges(
    "agent",
    should_continue,  # This function returns "action" or END
    {
        "action": "action",
        END: END  # <-- The agent's final answer is the end of the graph
    }
)
# The action loop is the same
graph.add_edge("action", "agent")

# 4. Compile and export
app = graph.compile()

 