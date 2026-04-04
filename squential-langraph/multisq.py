# Imports
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict
import torch


# -------------------------------
# Define State Schema
# -------------------------------
class LLMBlog(TypedDict):
    title: str
    outline: str
    blog: str


# -------------------------------
# Node 1: Generate Outline
# -------------------------------
def outline_gen(state: LLMBlog) -> dict:

    title = state["title"]

    # Load local Ollama model
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )

    # Check GPU availability
    device = torch.cuda.is_available()
    print("GPU Available:", device)

    prompt = f"Generate an outline for a blog titled: {title}"

    outline = llm.invoke(prompt).content

    return {"outline": outline}


# -------------------------------
# Node 2: Generate Blog
# -------------------------------
def blog_gen(state: LLMBlog) -> dict:

    title = state["title"]
    outline = state["outline"]

    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )

    prompt = f"""
    Write a detailed blog.

    Title: {title}

    Outline:
    {outline}
    """

    blog = llm.invoke(prompt).content

    return {"blog": blog}


# -------------------------------
# Build Graph
# -------------------------------
workflow = StateGraph(LLMBlog)

workflow.add_node("llm_outline", outline_gen)
workflow.add_node("llm_blog", blog_gen)

workflow.add_edge(START, "llm_outline")
workflow.add_edge("llm_outline", "llm_blog")
workflow.add_edge("llm_blog", END)

graph = workflow.compile()


# -------------------------------
# Run Graph
# -------------------------------
initial_state = {
    "title": "Rise of AI"
}

result = graph.invoke(initial_state)

print("\nFINAL RESULT:\n")
print(result)