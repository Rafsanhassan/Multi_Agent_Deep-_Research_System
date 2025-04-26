import os
from typing import Dict, List, Tuple, Any, Annotated
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.tavily_search import TavilySearchResults

import langgraph as lg
from langgraph.graph import END, StateGraph

# Configuration
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# Define state for the graph
class AgentState(BaseModel):
    """State for the research agent system."""
    query: str = Field(description="The original research query")
    research_findings: List[Dict] = Field(default_factory=list, description="Research findings from various sources")
    draft_answer: str = Field(default="", description="The current draft answer")
    final_answer: str = Field(default="", description="The final answer to present to user")
    source_documents: List[Dict] = Field(default_factory=list, description="Source documents and references")
    needs_more_research: bool = Field(default=True, description="Whether more research is needed")
    research_topics: List[str] = Field(default_factory=list, description="Specific topics to research further")
    
# Define node transition states
class AgentTransition(str, Enum):
    """Transitions between states in the system."""
    CONTINUE_RESEARCH = "continue_research"
    DRAFT_ANSWER = "draft_answer"
    REFINE_ANSWER = "refine_answer"
    COMPLETE = "complete"

# 1. Initialize Tools and Models
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# Tavily search tool
tavily_tool = Tool(
    name="web_search",
    description="Search the web for information on a specific query or topic",
    func=TavilySearchResults(max_results=5).invoke
)

# Vector store for research findings
vector_store = Chroma(
    collection_name="research_findings",
    embedding_function=embeddings
)

# 2. Research Agent
research_system_prompt = """You are an expert research agent. Your job is to:
1. Search the web for reliable information on the given topic
2. Extract key findings and organize them in a structured way
3. Identify gaps in research that need further investigation
4. Always cite your sources

Be thorough, accurate, and focus on finding information from diverse, reliable sources."""

research_prompt = ChatPromptTemplate.from_messages([
    ("system", research_system_prompt),
    ("human", "Research the following topic: {query}\nFocus on: {research_topics}\nCurrent findings: {research_findings}")
])

class ResearchOutput(BaseModel):
    """Output from research agent."""
    new_findings: List[Dict] = Field(description="New findings from research")
    new_research_topics: List[str] = Field(description="New topics identified for further research") 
    sources: List[Dict] = Field(description="Sources for the new findings")
    research_complete: bool = Field(description="Whether research is complete")

research_parser = JsonOutputParser(pydantic_object=ResearchOutput)

# Research agent chain
def research_agent(state: AgentState) -> Dict:
    """Research agent to gather information from the web."""
    # Format research prompt
    research_input = {
        "query": state.query,
        "research_topics": state.research_topics if state.research_topics else [state.query],
        "research_findings": state.research_findings
    }
    
    # Execute search with Tavily
    search_results = []
    for topic in (state.research_topics if state.research_topics else [state.query]):
        results = tavily_tool.invoke(topic)
        search_results.extend(results)
    
    # Have LLM process the results
    research_chain = research_prompt | llm | research_parser
    research_output = research_chain.invoke(research_input)
    
    # Update state
    new_findings = research_output.get("new_findings", [])
    new_sources = research_output.get("sources", [])
    new_topics = research_output.get("new_research_topics", [])
    research_complete = research_output.get("research_complete", False)
    
    return {
        "research_findings": state.research_findings + new_findings,
        "source_documents": state.source_documents + new_sources,
        "research_topics": new_topics,
        "needs_more_research": not research_complete
    }

# 3. Synthesis Agent
synthesis_system_prompt = """You are an expert synthesis agent. Your job is to:
1. Analyze all research findings provided
2. Create a well-structured, comprehensive draft answer
3. Ensure all claims are supported by the research
4. Cite sources appropriately
5. Identify any gaps that require additional research

Be clear, concise, and focus on providing an authoritative answer based solely on the provided research."""

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", synthesis_system_prompt),
    ("human", "Create a draft answer for: {query}\nUsing these research findings: {research_findings}\nCurrent draft (if any): {draft_answer}")
])

class SynthesisOutput(BaseModel):
    """Output from synthesis agent."""
    draft_answer: str = Field(description="Draft answer synthesized from research")
    missing_information: List[str] = Field(description="Missing information needed for a complete answer")
    
synthesis_parser = JsonOutputParser(pydantic_object=SynthesisOutput)

# Synthesis agent chain
def synthesis_agent(state: AgentState) -> Dict:
    """Synthesis agent to draft answers from research."""
    # Format synthesis prompt
    synthesis_input = {
        "query": state.query,
        "research_findings": state.research_findings,
        "draft_answer": state.draft_answer
    }
    
    # Generate draft
    synthesis_chain = synthesis_prompt | llm | synthesis_parser
    synthesis_output = synthesis_chain.invoke(synthesis_input)
    
    # Extract new research topics from missing information
    missing_info = synthesis_output.get("missing_information", [])
    
    return {
        "draft_answer": synthesis_output.get("draft_answer", state.draft_answer),
        "research_topics": missing_info if missing_info else [],
        "needs_more_research": len(missing_info) > 0
    }

# 4. Answer Refiner
refine_system_prompt = """You are an expert answer refiner. Your job is to:
1. Review the draft answer and research findings
2. Polish the answer to ensure it's comprehensive, well-structured, and accurate
3. Add proper citations and references
4. Make sure the final answer directly addresses the original query
5. Format the answer professionally with clear sections and organization

Produce a final, publication-quality answer."""

refine_prompt = ChatPromptTemplate.from_messages([
    ("system", refine_system_prompt),
    ("human", "Refine this draft answer for: {query}\nDraft: {draft_answer}\nEnsure all information from the research is included: {research_findings}\nInclude these sources: {source_documents}")
])

def refine_answer(state: AgentState) -> Dict:
    """Refine the draft answer into a final answer."""
    # Format refine prompt
    refine_input = {
        "query": state.query,
        "draft_answer": state.draft_answer,
        "research_findings": state.research_findings,
        "source_documents": state.source_documents
    }
    
    # Generate final answer
    refine_chain = refine_prompt | llm
    final_answer = refine_chain.invoke(refine_input)
    
    return {
        "final_answer": final_answer.content,
        "needs_more_research": False
    }

# 5. Routing Logic
def should_continue_research(state: AgentState) -> str:
    """Determine the next step in the research process."""
    if state.needs_more_research and len(state.research_findings) < 15:
        return AgentTransition.CONTINUE_RESEARCH
    elif not state.draft_answer:
        return AgentTransition.DRAFT_ANSWER
    elif state.needs_more_research and len(state.research_findings) >= 15:
        # Force drafting if we've done a lot of research
        return AgentTransition.DRAFT_ANSWER
    else:
        return AgentTransition.REFINE_ANSWER

# 6. Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", research_agent)
workflow.add_node("synthesize", synthesis_agent)
workflow.add_node("refine", refine_answer)

# Add edges
workflow.add_edge("research", should_continue_research)
workflow.add_conditional_edges(
    "research",
    should_continue_research,
    {
        AgentTransition.CONTINUE_RESEARCH: "research",
        AgentTransition.DRAFT_ANSWER: "synthesize"
    }
)

workflow.add_conditional_edges(
    "synthesize",
    lambda state: AgentTransition.CONTINUE_RESEARCH if state.needs_more_research else AgentTransition.REFINE_ANSWER,
    {
        AgentTransition.CONTINUE_RESEARCH: "research",
        AgentTransition.REFINE_ANSWER: "refine"
    }
)

workflow.add_edge("refine", END)

# 7. Create executable graph
graph = workflow.compile()

# 8. Main Research Function
def execute_deep_research(query: str):
    """Execute the deep research system on a given query."""
    print(f"Starting deep research on: {query}")
    
    # Initial state
    initial_state = AgentState(
        query=query,
        research_findings=[],
        draft_answer="",
        final_answer="",
        source_documents=[],
        needs_more_research=True,
        research_topics=[query]
    )
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    print(f"Research complete. Final answer generated.")
    return {
        "final_answer": result.final_answer,
        "sources": result.source_documents,
        "research_findings": result.research_findings
    }

# Example usage
if __name__ == "__main__":
    result = execute_deep_research("What are the latest developments in quantum computing and their potential impact on cryptography?")
    print("\n\nFINAL ANSWER:")
    print(result["final_answer"])
    print("\n\nSOURCES:")
    for source in result["sources"]:
        print(f"- {source.get('title', 'Untitled')}: {source.get('url', 'No URL')}")