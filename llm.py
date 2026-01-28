# src/genai/llm.py
"""
LLM Module - Using Groq API for all LLM operations.
Provides:
- Agent Brain: For agentic reasoning and tool calling
- RAG Summarizer: For document-based Q&A
"""

import os
import os.path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
from .retrivel.pince_ret import query_pinecone

load_dotenv()

# ---------------------------------------------------------------
# GROQ MODEL CONFIGURATION
# ---------------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️ Warning: GROQ_API_KEY not found in environment variables.")


# ---------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------

def get_groq_llm(model_name: str = None, temperature: float = 0):
    """
    Creates a ChatGroq instance with the specified model.
    
    Args:
        model_name: The Groq model to use. Defaults to GROQ_MODEL env var.
        temperature: Controls randomness (0 = deterministic, 1 = creative)
    
    Returns:
        ChatGroq: Configured Groq LLM instance
    """
    if model_name is None:
        model_name = GROQ_MODEL
    
    return ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=None,
        timeout=120,
        max_retries=2,
    )


def get_agent_brain(model_name: str = None):
    """
    Initializes the primary "brain" LLM for agentic reasoning.
    Uses Groq API with openai/gpt-oss-120b model.
    
    Args:
        model_name: Optional model override. Defaults to GROQ_MODEL.
    
    Returns:
        ChatGroq: LLM configured for agent operations
    """
    if model_name is None:
        model_name = os.getenv("AGENT_BRAIN_MODEL", GROQ_MODEL)
    
    return get_groq_llm(model_name=model_name, temperature=0)


def get_summarizer_llm():
    """
    Returns the ChatGroq instance for RAG summarization.
    Uses the same Groq model but can be configured separately if needed.
    
    Returns:
        ChatGroq: LLM configured for summarization
    """
    model_name = os.getenv("SUMMARIZER_MODEL", GROQ_MODEL)
    return get_groq_llm(model_name=model_name, temperature=0)


# ---------------------------------------------------------------
# INITIALIZE THE RE-RANKER MODEL
# ---------------------------------------------------------------
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-large'
try:
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
except Exception as e:
    print(f"Error initializing re-ranker model: {e}")
    reranker = None


# ---------------------------------------------------------------
# RAG FUNCTION (with Re-ranking and Citations)
# ---------------------------------------------------------------

def query_ollama_rag(query: str, namespace: str, top_k: int = 3, rerank_top_n: int = 20):
    """
    Retrieve, RE-RANK, and summarize relevant chunks using Groq.
    Appends a formatted list of sources to the final answer.
    
    Note: Function name kept as 'query_ollama_rag' for backward compatibility,
    but now uses Groq API instead of Ollama.
    """
    if reranker is None:
        return "Error: Re-ranker model is not available. Please check installation."

    results = query_pinecone(query, namespace, top_k=rerank_top_n)
    if not results:
        return "No relevant documents found in Pinecone."

    rerank_pairs = [[query, r['text']] for r in results]
    scores = reranker.predict(rerank_pairs)
    for i in range(len(results)):
        results[i]['rerank_score'] = scores[i]
    reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    final_results = reranked_results[:top_k]

    context = "\n\n".join([
        f"Source: {r['source']}\nPage: {r['page']}\nText:\n{r['text']}" for r in final_results
    ])

    system_prompt = """You are a financial analysis assistant. Your role is to provide EXTREMELY CONCISE and direct answers based *only* on the provided context.

INSTRUCTIONS:
1. ANSWER LIMIT: 3-4 SENTENCES MAX. Be direct. No fluff.
2. Base your answer strictly on the text provided in the CONTEXT.
3. If the context does not contain the answer, reply with: "The provided documents do not contain this information."
4. FOLLOW FORMATTING:
   - Use headings (##) and bullet points (`- `) for lists.
   - NO double asterisks (**text**) for bolding. Plain text only.
   - NO LaTeX. Explain math simply.
   - Tables: Text left-aligned, numbers right-aligned."""

    user_prompt = f"""CONTEXT:
{context}

---
QUESTION:
{query}

---
FINAL ANSWER:"""

    try:
        # Use Groq LLM for summarization
        summarizer_llm = get_summarizer_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = summarizer_llm.invoke(messages)
        summary_text = response.content

        # Build references
        sources_used = {}
        for r in final_results:
            file_name = os.path.basename(r['source']) 
            source_key = f"{file_name} (Page: {r['page']})"
            sources_used[source_key] = True
        
        reference_list = "\n".join([f"- {key}" for key in sources_used.keys()])
        
        final_answer = f"""{summary_text.strip()}

---
## References
{reference_list}
"""
        return final_answer

    except Exception as e:
        return f"Groq API error: {e}"
