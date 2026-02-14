from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from retrieval import get_query_engine
from config import *
import os
from config import OPENAI_API_KEY
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)
search_tool = TavilySearchResults(
    max_results=5,
    tavily_api_key=TAVILY_API_KEY
)


def rag_query_tool(query: str):
    engine = get_query_engine()
    return str(engine.query(query))


rag_tool = Tool(
    name="Document_RAG",
    func=rag_query_tool,
    description="Answer questions using uploaded documents"
)


def create_agent():
    tools = [rag_tool, search_tool]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful multi-agent RAG assistant. "
            "Use Document_RAG for document-based queries. "
            "Use Tavily search for internet or real-time questions."
        ),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )